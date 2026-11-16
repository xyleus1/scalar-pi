"""GitHub-hosted deadline cleanup. This program has no instance-launch operation.

The launcher must verify the live armed artifact AND the running Watch deadline
step before launching, then recheck launch_not_after_unix immediately before POST.
Only a separate expiring instance-read/terminate service key belongs here.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import time

from cloud import RESEARCH_PROFILE
from cloud import LambdaApiError
from cloud import LambdaClient

WORKER_COMMIT = "f53255a5965153172149546dc8eee95b2bd9b076"
REPOSITORY = "xyleus1/scalar-pi"
CLEANUP_SECONDS = 1200
POLL_SECONDS = 10
RECONCILE_SECONDS = 60
MAX_TERMINATE_ATTEMPTS = 3
JOB_TIMEOUT_SECONDS = 300 * 60
EVIDENCE_RESERVE_SECONDS = 300


def sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def github_identity():
    names = {
        "repository": "GITHUB_REPOSITORY",
        "run_id": "GITHUB_RUN_ID",
        "run_attempt": "GITHUB_RUN_ATTEMPT",
        "head_sha": "GITHUB_SHA",
        "ref": "GITHUB_REF",
    }
    identity = {key: os.environ.get(variable, "") for key, variable in names.items()}
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted":
        raise ValueError("Run this controller on a GitHub-hosted Actions runner")
    if identity["repository"] != REPOSITORY or identity["ref"] != "refs/heads/main":
        raise ValueError("Only the trusted repository main branch may arm cleanup")
    if not re.fullmatch(r"[a-f0-9]{40}", identity["head_sha"]):
        raise ValueError("Missing immutable controller commit")
    if not identity["run_id"].isdigit() or not identity["run_attempt"].isdigit():
        raise ValueError("Invalid workflow run identity")
    return identity


def validate_request(value, now):
    required = {"instance_name", "workspace_id", "worker_commit", "region", "instance_type", "deadline_utc"}
    if set(value) - required - {"launch_window_seconds"} or not required.issubset(value):
        raise ValueError("Request fields do not match the watchdog contract")
    if not re.fullmatch(r"scalar-pi-[a-f0-9]{12}", value["instance_name"]):
        raise ValueError("Use a fresh scalar-pi name with twelve random hexadecimal digits")
    if not re.fullmatch(r"(?:[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})", value["workspace_id"]):
        raise ValueError("Invalid workspace ID")
    if value["worker_commit"] != WORKER_COMMIT:
        raise ValueError("This watcher is bound to the qualified scientific worker commit")
    if (value["instance_type"], value["region"]) != RESEARCH_PROFILE:
        raise ValueError("Instance type and region must match the research profile")
    parsed = dt.datetime.fromisoformat(value["deadline_utc"].replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0) or parsed.microsecond:
        raise ValueError("Deadline must be an absolute UTC time in whole seconds")
    deadline = int(parsed.timestamp())
    window = value.get("launch_window_seconds", 600)
    if isinstance(window, bool) or not isinstance(window, int) or not 60 <= window <= 900:
        raise ValueError("Launch window must be between sixty and nine hundred seconds")
    if not window + 60 <= deadline - now <= 4 * 3600:
        raise ValueError("Deadline must allow the launch window and be at most four hours away")
    return {**value, "deadline_utc": parsed.isoformat(), "launch_window_seconds": window}, deadline


def tags(instance):
    values = instance.get("tags", [])
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError("Instance tags have an unexpected shape")
    result = {}
    for item in values:
        key = item.get("key")
        if not isinstance(key, str) or key in result:
            raise ValueError("Instance tag keys must be unique strings")
        result[key] = item.get("value")
    return result


def validate_instance(instance, expected, instance_id=None):
    if not isinstance(instance, dict) or not re.fullmatch(r"[a-f0-9-]{32,36}", instance.get("id", "")):
        raise ValueError("Invalid instance response")
    if instance_id is not None and instance["id"] != instance_id:
        raise ValueError("Bound instance ID changed")
    actual = {
        "instance_name": instance.get("name"),
        "workspace_id": instance.get("workspace_id"),
        "region": instance.get("region", {}).get("name"),
        "instance_type": instance.get("instance_type", {}).get("name"),
    }
    if any(actual[key] != expected[key] for key in actual):
        raise ValueError("Instance identity differs from the armed request")
    actual_tags = tags(instance)
    if any(actual_tags.get(key) != value for key, value in expected["tags"].items()):
        raise ValueError("Instance project, commit, run, or deadline tags differ from the armed request")
    return instance["id"]


def same_name(instances, name):
    if not isinstance(instances, list):
        raise ValueError("Instance listing has an unexpected shape")
    matches = [item for item in instances if isinstance(item, dict) and item.get("name") == name]
    if len(matches) > 1:
        raise ValueError("Duplicate expected instance names; cleanup will not choose one")
    return matches


def transient(error):
    if isinstance(error, LambdaApiError):
        return error.status in {408, 429} or 500 <= error.status < 600
    return isinstance(error, OSError | TimeoutError | RuntimeError)


def get(client, endpoint, until, *, now=time.time, sleep=time.sleep, missing_ok=False):
    while now() < until:
        try:
            return client.request("GET", endpoint)
        except (LambdaApiError, OSError, TimeoutError, RuntimeError) as error:
            if missing_ok and isinstance(error, LambdaApiError) and error.status == 404:
                return None
            if not transient(error):
                raise
            sleep(min(POLL_SECONDS, max(0, until - now())))
    raise TimeoutError("Read-only API retries exhausted the absolute cleanup deadline")


def arm(request, output, *, execute=False, client=None, now=time.time, sleep=time.sleep):
    identity = github_identity()
    observed = now()
    request, deadline = validate_request(request, observed)
    job_started = int(os.environ["WATCHDOG_JOB_STARTED_UNIX"])
    if (
        job_started > observed
        or deadline + CLEANUP_SECONDS + EVIDENCE_RESERVE_SECONDS > job_started + JOB_TIMEOUT_SECONDS
    ):
        raise ValueError("Cleanup and evidence publication do not fit the Actions job time limit")
    expected = {
        key: request[key] for key in ("instance_name", "workspace_id", "worker_commit", "region", "instance_type")
    }
    expected["tags"] = {
        "project": "scalar-pi",
        "commit": request["worker_commit"],
        "watchdog-run": identity["run_id"] + "-" + identity["run_attempt"],
        "watchdog-deadline": str(deadline),
    }
    if output.exists():
        raise ValueError("Refusing to overwrite an existing armed receipt")
    if execute:
        client = client or LambdaClient(os.environ.get("LAMBDA_TERMINATE_API_KEY"))
        instances = get(client, "instances", observed + 120, now=now, sleep=sleep)
        if same_name(instances, request["instance_name"]):
            raise ValueError("Expected instance name already exists; arm before launching")
    if now() >= observed + request["launch_window_seconds"]:
        raise TimeoutError("Arming consumed the launch window")
    receipt = {
        "schema_version": 1,
        "kind": "lambda-deadline-watchdog",
        "status": "armed",
        "dry_run": not execute,
        "armed_unix": observed,
        "launch_not_after_unix": observed + request["launch_window_seconds"],
        "deadline_unix": deadline,
        "cleanup_deadline_unix": deadline + CLEANUP_SECONDS,
        "job_started_unix": job_started,
        "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        "evidence_reserve_seconds": EVIDENCE_RESERVE_SECONDS,
        "request": request,
        "request_sha256": sha256(request),
        "expected": expected,
        "github": identity,
        "launch_gate_requires_running_step": "Watch deadline",
    }
    write_json(output, receipt)
    return receipt


def validate_receipt(receipt, *, execute, now):
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "lambda-deadline-watchdog":
        raise ValueError("Invalid armed receipt")
    if receipt.get("status") != "armed" or receipt.get("dry_run") is not (not execute):
        raise ValueError("Live and dry-run receipts cannot be interchanged")
    if receipt["github"] != github_identity() or sha256(receipt["request"]) != receipt["request_sha256"]:
        raise ValueError("Armed receipt belongs to different code, workflow run, or request")
    request, deadline = validate_request(receipt["request"], receipt["armed_unix"])
    expected_tags = {
        "project": "scalar-pi",
        "commit": request["worker_commit"],
        "watchdog-run": receipt["github"]["run_id"] + "-" + receipt["github"]["run_attempt"],
        "watchdog-deadline": str(deadline),
    }
    expected = {
        key: request[key] for key in ("instance_name", "workspace_id", "worker_commit", "region", "instance_type")
    }
    expected["tags"] = expected_tags
    if receipt["expected"] != expected or receipt["deadline_unix"] != deadline:
        raise ValueError("Armed identity or deadline was changed")
    if receipt["cleanup_deadline_unix"] != deadline + CLEANUP_SECONDS:
        raise ValueError("Cleanup deadline was extended")
    if receipt["job_started_unix"] != int(os.environ["WATCHDOG_JOB_STARTED_UNIX"]):
        raise ValueError("Armed receipt belongs to another job execution window")
    if (
        receipt["job_timeout_seconds"] != JOB_TIMEOUT_SECONDS
        or receipt["evidence_reserve_seconds"] != EVIDENCE_RESERVE_SECONDS
    ):
        raise ValueError("Job timeout or evidence reserve was changed")
    if receipt["cleanup_deadline_unix"] + EVIDENCE_RESERVE_SECONDS > receipt["job_started_unix"] + JOB_TIMEOUT_SECONDS:
        raise ValueError("Cleanup exceeds the Actions job execution window")
    if receipt["launch_not_after_unix"] != receipt["armed_unix"] + request["launch_window_seconds"]:
        raise ValueError("Launch window was extended")
    if now >= receipt["launch_not_after_unix"]:
        raise TimeoutError("The watcher did not start within the launch window; do not launch")


def watch(receipt, output, *, execute=False, client=None, now=time.time, sleep=time.sleep, dry_run_seconds=90):
    validate_receipt(receipt, execute=execute, now=now())
    if output.exists():
        raise ValueError("Refusing to overwrite an existing watchdog result")
    events = output.with_suffix(".events.jsonl")

    def event(status, **details):
        record = {"observed_unix": now(), "status": status, **details}
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    def finish(status, **details):
        result = {
            "schema_version": 1,
            "status": status,
            "dry_run": not execute,
            "request_sha256": receipt["request_sha256"],
            "github": receipt["github"],
            "observed_unix": now(),
            **details,
        }
        write_json(output, result)
        event(status, **details)
        return result

    output.parent.mkdir(parents=True, exist_ok=True)
    event("watching", launch_not_after_unix=receipt["launch_not_after_unix"], deadline_unix=receipt["deadline_unix"])
    if not execute:
        if not 0 <= dry_run_seconds <= 120:
            raise ValueError("Dry-run observation is limited to two minutes")
        until = now() + dry_run_seconds
        while now() < until:
            sleep(min(POLL_SECONDS, until - now()))
        return finish("dry_run_completed", api_requests=0)
    client = client or LambdaClient(os.environ.get("LAMBDA_TERMINATE_API_KEY"))
    expected = receipt["expected"]
    instance_id = None
    acknowledged = False
    attempts = 0
    last_post = None
    available_observations = 0
    try:
        # Keep discovering through cleanup expiry even after the admission window
        # closes: an accepted launch may become visible late. The launch gate,
        # not absence in an eventually consistent list, enforces launch admission.
        while now() < receipt["cleanup_deadline_unix"]:
            until = receipt["cleanup_deadline_unix"]
            if instance_id is None:
                matches = same_name(get(client, "instances", until, now=now, sleep=sleep), expected["instance_name"])
                if not matches:
                    sleep(min(POLL_SECONDS, max(0, until - now())))
                    continue
                instance_id = validate_instance(matches[0], expected)
                write_json(
                    output.with_name("binding.json"),
                    {
                        "instance_id": instance_id,
                        "request_sha256": receipt["request_sha256"],
                        "bound_unix": now(),
                        "expected": expected,
                    },
                )
                event("bound", instance_id=instance_id)
            instance = get(client, "instances/" + instance_id, until, now=now, sleep=sleep, missing_ok=True)
            if instance is None:
                return finish(
                    "terminated", instance_id=instance_id, confirmation="bound_id_404", terminate_attempts=attempts
                )
            if not isinstance(instance, dict) or instance.get("id") != instance_id:
                raise ValueError("Bound instance ID changed")
            if instance.get("status") == "terminated":
                return finish(
                    "terminated", instance_id=instance_id, confirmation="provider_status", terminate_attempts=attempts
                )
            if instance.get("status") == "terminating":
                acknowledged = True
            if not acknowledged:
                validate_instance(instance, expected, instance_id)
            if receipt["deadline_unix"] <= now() < until and not acknowledged:
                available = instance.get("actions", {}).get("terminate", {}).get("available") is True
                available_observations = available_observations + 1 if available else 0
                reconciled = last_post is None or (
                    now() - last_post >= RECONCILE_SECONDS and available_observations >= 3
                )
                if available and reconciled and attempts < MAX_TERMINATE_ATTEMPTS:
                    attempts += 1
                    last_post = now()
                    available_observations = 0
                    event("terminate_requested", instance_id=instance_id, attempt=attempts)
                    try:
                        response = client.request(
                            "POST", "instance-operations/terminate", {"instance_ids": [instance_id]}
                        )
                    except (LambdaApiError, OSError, TimeoutError, RuntimeError) as error:
                        if not transient(error):
                            raise
                        event("terminate_response_uncertain", error_type=type(error).__name__)
                    else:
                        accepted = response.get("terminated_instances", []) if isinstance(response, dict) else []
                        acknowledged = any(
                            isinstance(item, dict) and item.get("id") == instance_id for item in accepted
                        )
                        event("terminate_acknowledged" if acknowledged else "terminate_response_uncertain")
            sleep(min(POLL_SECONDS, max(0, until - now())))
        return finish(
            "unresolved" if instance_id else "not_observed",
            instance_id=instance_id,
            terminate_attempts=attempts,
            termination_acknowledged=acknowledged,
        )
    except Exception as error:
        finish(
            "failed",
            instance_id=instance_id,
            error_type=type(error).__name__,
            http_status=error.status if isinstance(error, LambdaApiError) else None,
            terminate_attempts=attempts,
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("arm", "watch"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--armed", type=Path)
    parser.add_argument("--execute", action="store_true", help="Use the scoped Lambda key; otherwise no cloud requests")
    parser.add_argument("--dry-run-seconds", type=int, default=90)
    args = parser.parse_args(argv)
    try:
        if args.mode == "arm":
            arm(json.loads(os.environ["SCALAR_PI_WATCHDOG_REQUEST"]), args.output, execute=args.execute)
            print("Armed receipt written. Launch still requires the running Watch deadline step.", flush=True)
            return 0
        if args.armed is None:
            parser.error("watch requires --armed")
        result = watch(
            json.loads(args.armed.read_text()), args.output, execute=args.execute, dry_run_seconds=args.dry_run_seconds
        )
        return int(result["status"] not in {"terminated", "dry_run_completed"})
    except Exception as error:
        # Never emit arbitrary provider response bodies or exception strings.
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
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
