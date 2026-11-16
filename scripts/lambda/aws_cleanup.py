"""Exact-instance deadline cleanup for AWS Lambda and EventBridge Scheduler.

Deploy with cloud.py and watchdog.py in the same ZIP. Only the separately
scoped Lambda Cloud read/terminate credential is read from Secrets Manager.
The immutable function version binds the lease specification; DynamoDB keeps
POST intent and acknowledgment across duplicate deliveries and runtime exits.
"""

import hashlib
import json
import os
import re
import time
import uuid

from cloud import LambdaApiError
from cloud import LambdaClient
from watchdog import MAX_TERMINATE_ATTEMPTS
from watchdog import POLL_SECONDS
from watchdog import RECONCILE_SECONDS
from watchdog import WORKER_COMMIT
from watchdog import sha256
from watchdog import validate_instance

MAX_LEASE_SECONDS = 24 * 3600
CLEANUP_SECONDS = 2 * 3600
INVOCATION_SECONDS = 12 * 60
ID_PATTERN = r"(?:[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})"


def validate_spec(spec, expected_hash, lease_id):
    fields = {
        "schema_version",
        "lease_id",
        "instance_id",
        "expected",
        "issued_unix",
        "deadline_unix",
        "cleanup_deadline_unix",
        "key_expires_unix",
        "secret_version_id",
        "cleanup_key_sha256",
    }
    if not isinstance(spec, dict) or set(spec) != fields or spec["schema_version"] != 1:
        raise ValueError("Unsupported cleanup lease specification")
    if sha256(spec) != expected_hash or spec["lease_id"] != lease_id or not re.fullmatch(r"[a-f0-9]{32}", lease_id):
        raise ValueError("Cleanup lease identity or hash differs")
    if not isinstance(spec["instance_id"], str) or not re.fullmatch(ID_PATTERN, spec["instance_id"]):
        raise ValueError("Cleanup requires an exact instance ID")
    expected = spec["expected"]
    if not isinstance(expected, dict) or set(expected) != {
        "instance_name",
        "workspace_id",
        "worker_commit",
        "region",
        "instance_type",
        "tags",
    }:
        raise ValueError("Cleanup requires the complete expected instance identity")
    if not re.fullmatch(r"scalar-pi-[a-f0-9]{12}", expected["instance_name"]):
        raise ValueError("Invalid expected instance name")
    if not re.fullmatch(ID_PATTERN, expected["workspace_id"]) or expected["worker_commit"] != WORKER_COMMIT:
        raise ValueError("Invalid expected workspace or scientific commit")
    if any(not isinstance(expected[key], str) or not expected[key] for key in ("region", "instance_type")):
        raise ValueError("Expected region and instance type are required")
    tags = expected["tags"]
    if (
        not isinstance(tags, dict)
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in tags.items())
        or tags.get("project") != "scalar-pi"
        or tags.get("commit") != WORKER_COMMIT
        or not re.fullmatch(r"\d+-\d+", tags.get("watchdog-run", ""))
        or not tags.get("watchdog-deadline", "").isdigit()
    ):
        raise ValueError("Original project, commit and watchdog tags are required")
    for key in ("issued_unix", "deadline_unix", "cleanup_deadline_unix", "key_expires_unix"):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError("Lease times must be positive integer UTC timestamps")
    if not 0 < spec["deadline_unix"] - spec["issued_unix"] <= MAX_LEASE_SECONDS:
        raise ValueError("Cleanup lease must be finite and at most twenty-four hours")
    if spec["cleanup_deadline_unix"] != spec["deadline_unix"] + CLEANUP_SECONDS:
        raise ValueError("Cleanup retry window must be exactly two hours")
    if spec["key_expires_unix"] < spec["cleanup_deadline_unix"]:
        raise ValueError("Cleanup key must cover the complete retry window")
    if not isinstance(spec["secret_version_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9-]{32,64}", spec["secret_version_id"]
    ):
        raise ValueError("Pin the actual Secrets Manager version ID")
    if not re.fullmatch(r"[a-f0-9]{64}", spec["cleanup_key_sha256"]):
        raise ValueError("Pin the audited scoped cleanup key hash")
    return spec


def initial_state():
    return {
        "attempts": 0,
        "last_post_unix": None,
        "acknowledged": False,
        "terminated": False,
        "proofs": {},
    }


class LeaseBusyError(RuntimeError):
    pass


class CleanupIncompleteError(RuntimeError):
    pass


def conditional_failure(error):
    return getattr(error, "response", {}).get("Error", {}).get("Code") == "ConditionalCheckFailedException"


class DynamoState:
    """One item, strongly consistent reads and a conditional owner lease."""

    def __init__(self, client, table, lease_id, specification_hash):
        self.client, self.table = client, table
        self.key = {"lease_id": {"S": lease_id}}
        self.specification_hash = specification_hash
        self.owner, self.revision = None, None

    def acquire(self, owner, now, until):
        try:
            self.client.put_item(
                TableName=self.table,
                Item={
                    **self.key,
                    "spec_sha256": {"S": self.specification_hash},
                    "revision": {"N": "0"},
                    "state_json": {"S": json.dumps(initial_state())},
                },
                ConditionExpression="attribute_not_exists(lease_id)",
            )
        except Exception as error:
            if not conditional_failure(error):
                raise
        try:
            response = self.client.update_item(
                TableName=self.table,
                Key=self.key,
                UpdateExpression="SET lease_owner = :owner, owner_until = :until",
                ConditionExpression=(
                    "spec_sha256 = :hash AND " "(attribute_not_exists(owner_until) OR owner_until < :now)"
                ),
                ExpressionAttributeValues={
                    ":owner": {"S": owner},
                    ":until": {"N": str(until)},
                    ":now": {"N": str(now)},
                    ":hash": {"S": self.specification_hash},
                },
                ReturnValues="ALL_NEW",
            )
        except Exception as error:
            if conditional_failure(error):
                item = self.client.get_item(TableName=self.table, Key=self.key, ConsistentRead=True).get("Item", {})
                if item.get("spec_sha256", {}).get("S") != self.specification_hash:
                    raise ValueError("Stored cleanup lease specification differs") from None
                raise LeaseBusyError("Another invocation owns the cleanup lease") from None
            raise
        item = response["Attributes"]
        self.owner, self.revision = owner, int(item["revision"]["N"])
        return json.loads(item["state_json"]["S"])

    def save(self, state, now):
        self.client.update_item(
            TableName=self.table,
            Key=self.key,
            UpdateExpression="SET state_json = :state, revision = :next",
            ConditionExpression="spec_sha256 = :hash AND lease_owner = :owner AND owner_until >= :now AND revision = :revision",
            ExpressionAttributeValues={
                ":state": {"S": json.dumps(state, sort_keys=True)},
                ":next": {"N": str(self.revision + 1)},
                ":revision": {"N": str(self.revision)},
                ":hash": {"S": self.specification_hash},
                ":owner": {"S": self.owner},
                ":now": {"N": str(now)},
            },
        )
        self.revision += 1

    def release(self):
        if self.owner is not None:
            self.client.update_item(
                TableName=self.table,
                Key=self.key,
                UpdateExpression="REMOVE lease_owner, owner_until",
                ConditionExpression="lease_owner = :owner",
                ExpressionAttributeValues={":owner": {"S": self.owner}},
            )
            self.owner = None


def validate_event(event, spec, context):
    required = {"lease_id", "lease_sha256", "mode"}
    scheduled = {"schedule_arn", "scheduled_time", "execution_id"}
    if (
        not isinstance(event, dict)
        or set(event) not in (required, required | scheduled)
        or event["lease_id"] != spec["lease_id"]
        or event["lease_sha256"] != sha256(spec)
        or event["mode"] not in {"verify", "execute"}
    ):
        raise ValueError("Invocation does not match the immutable cleanup lease")
    if not str(context.function_version).isdigit() or int(context.function_version) < 1:
        raise ValueError("Invoke a published function version")
    if not context.invoked_function_arn.endswith(":" + str(context.function_version)):
        raise ValueError("Invocation must use its immutable version ARN")
    if scheduled.issubset(event):
        if any(not isinstance(event[key], str) or not event[key] for key in scheduled):
            raise ValueError("Incomplete scheduler delivery context")
        if not re.fullmatch(
            r"arn:aws:scheduler:[a-z0-9-]+:\d{12}:schedule/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", event["schedule_arn"]
        ):
            raise ValueError("Invalid scheduler ARN")
    elif event["mode"] == "execute":
        raise ValueError("Execution requires scheduler delivery context")


def transient(error):
    if isinstance(error, LambdaApiError):
        return error.status in {408, 429} or 500 <= error.status < 600
    return isinstance(error, OSError | TimeoutError | RuntimeError)


def cleanup(spec, event, context, store, client, *, now=time.time, sleep=time.sleep):
    """Read-only verification or bounded cleanup; unresolved execution raises."""
    validate_event(event, spec, context)
    started = int(now())
    if spec["issued_unix"] > started:
        raise ValueError("Cleanup lease issue time cannot be in the future")
    remaining = context.get_remaining_time_in_millis() / 1000
    if remaining < 90:
        raise CleanupIncompleteError("Insufficient invocation time to preserve cleanup evidence")
    if started >= spec["key_expires_unix"]:
        raise CleanupIncompleteError("Scoped cleanup credential expired")
    if event["mode"] == "execute" and started < spec["deadline_unix"]:
        raise ValueError("Cleanup cannot execute before its absolute deadline")
    if started >= spec["cleanup_deadline_unix"]:
        raise CleanupIncompleteError("Absolute cleanup retry window expired")
    until = min(started + INVOCATION_SECONDS, now() + remaining - 60, spec["cleanup_deadline_unix"])
    owner = uuid.uuid4().hex
    while now() < until and context.get_remaining_time_in_millis() > 60_000:
        try:
            state = store.acquire(owner, int(now()), int(now() + context.get_remaining_time_in_millis() / 1000 + 30))
            break
        except LeaseBusyError:
            # Early runtime exits may leave a fifteen-minute owner lease. Do not
            # consume both AWS retries immediately while the old lease expires.
            sleep(min(POLL_SECONDS, max(0, until - now())))
    else:
        raise CleanupIncompleteError("Owner lease still active; asynchronous retry required")
    observations = 0

    def persist():
        store.save(state, int(now()))

    def result(status, **details):
        return {"status": status, "lease_id": spec["lease_id"], "instance_id": spec["instance_id"], **details}

    try:
        if state["terminated"]:
            return result("terminated", confirmation="durable_receipt", terminate_attempts=state["attempts"])
        if event["mode"] == "execute":
            for kind in ("direct", "scheduled"):
                proof = state["proofs"].get(kind, {})
                if proof.get("function_arn") != context.invoked_function_arn or proof.get("lease_sha256") != sha256(
                    spec
                ):
                    raise ValueError("Both actual GET-only smoke proofs are required before cleanup handover")
        while now() < until and context.get_remaining_time_in_millis() > 60_000:
            try:
                instance = client.request("GET", "instances/" + spec["instance_id"])
            except (LambdaApiError, OSError, TimeoutError, RuntimeError) as error:
                if isinstance(error, LambdaApiError) and error.status == 404:
                    if not state["proofs"]:
                        raise ValueError("Cannot verify an absent instance") from None
                    instance = None
                elif transient(error):
                    sleep(min(POLL_SECONDS, max(0, until - now())))
                    continue
                else:
                    raise
            if instance is not None and (not isinstance(instance, dict) or instance.get("id") != spec["instance_id"]):
                raise ValueError("Bound instance ID changed")
            terminal = instance is None or instance.get("status") == "terminated"
            if event["mode"] == "verify":
                if terminal:
                    raise ValueError("Read-only smoke requires the current live instance")
                validate_instance(instance, spec["expected"], spec["instance_id"])
                proof = {
                    "observed_unix": int(now()),
                    "function_arn": context.invoked_function_arn,
                    "lease_sha256": sha256(spec),
                    "instance_id": spec["instance_id"],
                    **{key: event[key] for key in ("schedule_arn", "scheduled_time", "execution_id") if key in event},
                }
                kind = "scheduled" if "schedule_arn" in event else "direct"
                state["proofs"][kind] = proof
                persist()
                return result("verified", proof_kind=kind, proof=proof, terminate_attempts=state["attempts"])
            if terminal:
                state.update(terminated=True, confirmed_unix=int(now()))
                persist()
                return result(
                    "terminated",
                    confirmation="bound_id_404" if instance is None else "provider_status",
                    terminate_attempts=state["attempts"],
                )
            if instance.get("status") == "terminating" and not state["acknowledged"]:
                state["acknowledged"] = True
                persist()
            if not state["acknowledged"]:
                validate_instance(instance, spec["expected"], spec["instance_id"])
                available = instance.get("actions", {}).get("terminate", {}).get("available") is True
                observations = observations + 1 if available else 0
                last_post = state["last_post_unix"]
                reconciled = last_post is None or (now() - last_post >= RECONCILE_SECONDS and observations >= 3)
                if available and reconciled and state["attempts"] < MAX_TERMINATE_ATTEMPTS and now() < until:
                    state.update(attempts=state["attempts"] + 1, last_post_unix=int(now()))
                    persist()  # Write-ahead intent is not an acknowledgment.
                    observations = 0
                    if context.get_remaining_time_in_millis() <= 60_000 or now() >= until:
                        raise CleanupIncompleteError("Invocation ended after recording POST intent")
                    try:
                        response = client.request(
                            "POST", "instance-operations/terminate", {"instance_ids": [spec["instance_id"]]}
                        )
                    except (LambdaApiError, OSError, TimeoutError, RuntimeError) as error:
                        if not transient(error):
                            raise
                    else:
                        accepted = response.get("terminated_instances", []) if isinstance(response, dict) else []
                        if any(isinstance(item, dict) and item.get("id") == spec["instance_id"] for item in accepted):
                            state["acknowledged"] = True
                            persist()
            sleep(min(POLL_SECONDS, max(0, until - now())))
        raise CleanupIncompleteError("Termination remains unconfirmed; asynchronous retry required")
    finally:
        store.release()


def handler(event, context):
    """AWS entrypoint. Never log secret values or arbitrary provider responses."""
    try:
        import boto3
        from botocore.config import Config

        spec = validate_spec(
            json.loads(os.environ["SCALAR_PI_LEASE_JSON"]),
            os.environ["SCALAR_PI_LEASE_SHA256"],
            os.environ["SCALAR_PI_LEASE_ID"],
        )
        validate_event(event, spec, context)
        settings = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2, "mode": "standard"})
        secret = boto3.client("secretsmanager", config=settings).get_secret_value(
            SecretId=os.environ["SCALAR_PI_SECRET_ARN"],
            VersionId=spec["secret_version_id"],
        )
        key = secret["SecretString"]
        if (
            secret["VersionId"] != spec["secret_version_id"]
            or hashlib.sha256(key.encode()).hexdigest() != spec["cleanup_key_sha256"]
        ):
            raise ValueError("Cleanup credential differs from its audited version")
        store = DynamoState(
            boto3.client("dynamodb", config=settings),
            os.environ["SCALAR_PI_STATE_TABLE"],
            spec["lease_id"],
            sha256(spec),
        )
        outcome = cleanup(spec, event, context, store, LambdaClient(key))
        print(json.dumps(outcome, sort_keys=True), flush=True)
        return outcome
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "http_status": error.status if isinstance(error, LambdaApiError) else None,
                }
            ),
            flush=True,
        )
        raise RuntimeError(
            "Exact-instance cleanup did not complete; inspect the retained state and failure queue"
        ) from None
