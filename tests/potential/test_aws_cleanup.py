"""Offline AWS cleanup tests. No cloud account, credentials or GPU are used."""

import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[2]
FUNCTION_ARN = "arn:aws:lambda:us-east-1:123456789012:function:scalar-pi-cleanup:1"
SCHEDULE_ARN = "arn:aws:scheduler:us-east-1:123456789012:schedule/scalar-pi/cleanup"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/lambda"))
    spec = importlib.util.spec_from_file_location("scalar_pi_aws_cleanup_test", ROOT / "scripts/lambda/aws_cleanup.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class Clock:
    def __init__(self):
        self.value = 1800000000

    def now(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.value += seconds


class MemoryState:
    def __init__(self, module):
        self.state = module.initial_state()
        self.owner = None
        self.snapshots = []

    def acquire(self, owner, now, until):
        assert self.owner is None
        assert until > now
        self.owner = owner
        return copy.deepcopy(self.state)

    def save(self, state, now):
        assert self.owner is not None
        self.state = copy.deepcopy(state)
        self.snapshots.append((now, copy.deepcopy(state)))

    def release(self):
        self.owner = None


class FakeCloud:
    def __init__(self, spec, clock):
        self.clock = clock
        self.instance = {
            "id": spec["instance_id"],
            "name": spec["expected"]["instance_name"],
            "workspace_id": spec["expected"]["workspace_id"],
            "region": {"name": spec["expected"]["region"]},
            "instance_type": {"name": spec["expected"]["instance_type"]},
            "tags": [{"key": key, "value": value} for key, value in spec["expected"]["tags"].items()],
            "status": "active",
            "actions": {"terminate": {"available": True}},
        }
        self.calls = []
        self.get_effect = None
        self.post_effect = None
        self.terminated_at = None

    def request(self, method, endpoint, body=None):
        self.calls.append((self.clock.now(), method, endpoint, body))
        if method == "GET":
            assert endpoint == "instances/" + self.instance["id"]
            if self.get_effect:
                return self.get_effect()
            if self.terminated_at is not None and self.clock.now() >= self.terminated_at:
                return {"id": self.instance["id"], "status": "terminated"}
            return copy.deepcopy(self.instance)
        assert method == "POST"
        assert endpoint == "instance-operations/terminate"
        assert body == {"instance_ids": [self.instance["id"]]}
        if self.post_effect:
            return self.post_effect()
        self.instance["status"] = "terminating"
        self.terminated_at = self.clock.now() + 20
        return {"terminated_instances": [{"id": self.instance["id"]}]}

    def posts(self):
        return [call for call in self.calls if call[1] == "POST"]


@pytest.fixture
def case(module):
    clock = Clock()
    key = "fixture-scoped-key"
    spec = {
        "schema_version": 1,
        "lease_id": "a" * 32,
        "instance_id": "b" * 32,
        "expected": {
            "instance_name": "scalar-pi-012345abcdef",
            "workspace_id": "c" * 32,
            "worker_commit": module.WORKER_COMMIT,
            "region": "us-west-3",
            "instance_type": "gpu_1x_h100_pcie",
            "tags": {
                "project": "scalar-pi",
                "commit": module.WORKER_COMMIT,
                "watchdog-run": "123-1",
                "watchdog-deadline": str(clock.now() + 60),
            },
        },
        "issued_unix": clock.now(),
        "deadline_unix": clock.now() + 3600,
        "cleanup_deadline_unix": clock.now() + 3600 + module.CLEANUP_SECONDS,
        "key_expires_unix": clock.now() + 3600 + module.CLEANUP_SECONDS + 60,
        "secret_version_id": "d" * 32,
        "cleanup_key_sha256": hashlib.sha256(key.encode()).hexdigest(),
    }
    module.validate_spec(spec, module.sha256(spec), spec["lease_id"])
    store = MemoryState(module)
    client = FakeCloud(spec, clock)
    return SimpleNamespace(module=module, clock=clock, spec=spec, store=store, client=client, key=key)


def event(case, mode="verify", *, scheduled=False):
    value = {"lease_id": case.spec["lease_id"], "lease_sha256": case.module.sha256(case.spec), "mode": mode}
    if scheduled:
        value.update(
            schedule_arn=SCHEDULE_ARN, scheduled_time="2027-01-15T08:00:00Z", execution_id="scheduler-execution"
        )
    return value


def context(case):
    started = case.clock.now()
    return SimpleNamespace(
        function_version="1",
        invoked_function_arn=FUNCTION_ARN,
        get_remaining_time_in_millis=lambda: max(0, (900 - (case.clock.now() - started)) * 1000),
    )


def invoke(case, mode="execute", *, scheduled=True):
    return case.module.cleanup(
        case.spec,
        event(case, mode, scheduled=scheduled),
        context(case),
        case.store,
        case.client,
        now=case.clock.now,
        sleep=case.clock.sleep,
    )


@pytest.fixture
def ready(case):
    assert invoke(case, "verify", scheduled=False)["proof_kind"] == "direct"
    assert invoke(case, "verify", scheduled=True)["proof_kind"] == "scheduled"
    case.clock.value = case.spec["deadline_unix"]
    return case


def test_actual_get_only_smokes_preserve_both_proofs(case):
    direct = invoke(case, "verify", scheduled=False)
    scheduled = invoke(case, "verify", scheduled=True)
    assert direct["status"] == scheduled["status"] == "verified"
    assert set(case.store.state["proofs"]) == {"direct", "scheduled"}
    assert case.store.state["proofs"]["scheduled"]["schedule_arn"] == SCHEDULE_ARN
    assert len(case.client.calls) == 2
    assert not case.client.posts()


def test_exact_instance_terminates_and_duplicate_delivery_is_read_only(ready):
    result = invoke(ready)
    assert result["status"] == "terminated"
    assert ready.store.state["terminated"] is True
    assert ready.store.state["acknowledged"] is True
    assert len(ready.client.posts()) == 1
    before = len(ready.client.calls)
    assert invoke(ready)["confirmation"] == "durable_receipt"
    assert len(ready.client.calls) == before


@pytest.mark.parametrize("kind", ["direct", "scheduled"])
def test_both_smoke_proofs_are_required(ready, kind):
    del ready.store.state["proofs"][kind]
    with pytest.raises(ValueError, match="Both actual"):
        invoke(ready)
    assert not ready.client.posts()


def test_old_tag_deadline_is_not_the_replacement_operational_deadline(ready):
    assert int(ready.spec["expected"]["tags"]["watchdog-deadline"]) < ready.spec["deadline_unix"]
    ready.clock.value -= 1
    with pytest.raises(ValueError, match="before its absolute deadline"):
        invoke(ready)
    assert not ready.client.posts()


def test_future_lease_issue_time_cannot_extend_the_operational_window(case):
    case.clock.value -= 1
    with pytest.raises(ValueError, match="issue time"):
        invoke(case, "verify", scheduled=False)
    assert not case.client.calls


@pytest.mark.parametrize("field", ["name", "workspace_id", "region", "instance_type", "tags"])
def test_wrong_identity_never_terminates(ready, field):
    del ready.client.instance[field]
    with pytest.raises(ValueError, match="identity|tags"):
        invoke(ready)
    assert not ready.client.posts()


def test_changed_exact_id_is_rejected(ready):
    ready.client.get_effect = lambda: {"id": "e" * 32, "status": "terminated"}
    with pytest.raises(ValueError, match="ID changed"):
        invoke(ready)
    assert not ready.client.posts()


def test_bound_404_confirms_termination(ready):
    def missing():
        raise ready.module.LambdaApiError("GET", "instances/id", 404)

    ready.client.get_effect = missing
    assert invoke(ready)["confirmation"] == "bound_id_404"
    assert not ready.client.posts()


@pytest.mark.parametrize("status", ["active", "unhealthy", "booting"])
def test_acknowledgment_survives_function_retry_and_stale_status(ready, status):
    ready.client.instance["status"] = status
    ready.client.post_effect = lambda: {"terminated_instances": [{"id": ready.spec["instance_id"]}]}
    for _ in range(2):
        with pytest.raises(ready.module.CleanupIncompleteError, match="unconfirmed"):
            invoke(ready)
    assert len(ready.client.posts()) == 1
    assert ready.store.state["acknowledged"] is True


def test_terminating_status_is_sticky_across_invocations(ready):
    ready.client.instance["status"] = "terminating"
    with pytest.raises(ready.module.CleanupIncompleteError):
        invoke(ready)
    ready.client.instance["status"] = "unhealthy"
    with pytest.raises(ready.module.CleanupIncompleteError):
        invoke(ready)
    assert ready.store.state["acknowledged"] is True
    assert not ready.client.posts()


@pytest.mark.parametrize("status", ["active", "unhealthy", "booting"])
def test_uncertain_post_reconciles_and_lifetime_attempts_are_bounded(ready, status):
    ready.client.instance["status"] = status

    def lost():
        raise TimeoutError("Provider response lost")

    ready.client.post_effect = lost
    for _ in range(2):
        with pytest.raises(ready.module.CleanupIncompleteError):
            invoke(ready)
    posts = ready.client.posts()
    assert len(posts) == ready.module.MAX_TERMINATE_ATTEMPTS
    assert all(second[0] - first[0] >= 60 for first, second in itertools.pairwise(posts))
    assert ready.store.state["attempts"] == len(posts)


def test_crash_after_intent_before_post_reconciles_instead_of_assuming_ack(ready, monkeypatch):
    save = ready.store.save

    def crash(state, now):
        save(state, now)
        if state["attempts"] == 1:
            raise SystemExit("simulated runtime exit before POST")

    monkeypatch.setattr(ready.store, "save", crash)
    with pytest.raises(SystemExit):
        invoke(ready)
    assert ready.store.state["attempts"] == 1
    assert not ready.store.state["acknowledged"]
    assert not ready.client.posts()
    monkeypatch.setattr(ready.store, "save", save)
    assert invoke(ready)["status"] == "terminated"
    assert ready.client.posts()[0][0] >= ready.spec["deadline_unix"] + 60


def test_crash_after_provider_acceptance_reconciles_without_duplicate_post(ready):
    def accepted_then_crash():
        ready.client.instance["status"] = "terminating"
        ready.client.terminated_at = ready.clock.now() + 20
        raise SystemExit("simulated runtime exit after provider accepted")

    ready.client.post_effect = accepted_then_crash
    with pytest.raises(SystemExit):
        invoke(ready)
    assert not ready.store.state["acknowledged"]
    assert invoke(ready)["status"] == "terminated"
    assert len(ready.client.posts()) == 1


def test_transient_reads_exhaust_invocation_and_raise_for_async_retry(ready):
    def unavailable():
        raise ready.module.LambdaApiError("GET", "instances/id", 503)

    ready.client.get_effect = unavailable
    with pytest.raises(ready.module.CleanupIncompleteError):
        invoke(ready)
    assert ready.clock.now() == ready.spec["deadline_unix"] + ready.module.INVOCATION_SECONDS
    assert ready.store.owner is None
    assert not ready.client.posts()


def test_stale_owner_waits_through_actual_aws_retry_delays(ready, monkeypatch):
    acquire = ready.store.acquire
    original_start = ready.spec["deadline_unix"]

    def stale_owner(owner, now, until):
        if now <= original_start + 930:
            raise ready.module.LeaseBusyError("Earlier runtime exited without releasing ownership")
        return acquire(owner, now, until)

    monkeypatch.setattr(ready.store, "acquire", stale_owner)
    # The original invocation died early after claiming a 900-second runtime
    # plus 30-second ownership margin. AWS retries after one and two minutes.
    ready.clock.value = original_start + 60
    with pytest.raises(ready.module.CleanupIncompleteError, match="Owner lease"):
        invoke(ready)
    assert ready.clock.now() == original_start + 60 + ready.module.INVOCATION_SECONDS
    assert not ready.client.posts()
    ready.clock.sleep(120)
    assert invoke(ready)["status"] == "terminated"
    assert len(ready.client.posts()) == 1
    assert ready.client.posts()[0][0] > original_start + 930


def test_no_post_after_read_crosses_cleanup_window(ready):
    ready.clock.value = ready.spec["cleanup_deadline_unix"] - 5

    def slow():
        ready.clock.sleep(10)
        return copy.deepcopy(ready.client.instance)

    ready.client.get_effect = slow
    with pytest.raises(ready.module.CleanupIncompleteError):
        invoke(ready)
    assert not ready.client.posts()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deadline_unix", 1800000000 + 86401),
        ("cleanup_deadline_unix", 1800010801),
        ("key_expires_unix", 1800000100),
        ("issued_unix", True),
        ("instance_id", "other"),
        ("secret_version_id", "latest"),
        ("cleanup_key_sha256", "unknown"),
    ],
)
def test_lease_spec_rejects_unsafe_inputs(case, field, value):
    case.spec[field] = value
    with pytest.raises(ValueError, match="Cleanup|Lease|Pin"):
        case.module.validate_spec(case.spec, case.module.sha256(case.spec), case.spec["lease_id"])


def test_handler_redacts_provider_errors(case, monkeypatch, capsys):
    import sys

    monkeypatch.setenv("SCALAR_PI_LEASE_JSON", json.dumps(case.spec))
    monkeypatch.setenv("SCALAR_PI_LEASE_SHA256", case.module.sha256(case.spec))
    monkeypatch.setenv("SCALAR_PI_LEASE_ID", case.spec["lease_id"])
    monkeypatch.setenv("SCALAR_PI_STATE_TABLE", "fixture-table")
    monkeypatch.setenv("SCALAR_PI_SECRET_ARN", "fixture-secret-arn")
    secret = SimpleNamespace(
        get_secret_value=lambda **kwargs: {"VersionId": case.spec["secret_version_id"], "SecretString": case.key}
    )
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda name, **kwargs: secret if name == "secretsmanager" else object()),
    )
    monkeypatch.setitem(sys.modules, "botocore.config", SimpleNamespace(Config=lambda **kwargs: None))
    monkeypatch.setattr(case.module, "DynamoState", lambda *args: case.store)
    monkeypatch.setattr(case.module, "LambdaClient", lambda key: case.client)

    def failure():
        raise RuntimeError("must-not-log-" + case.key)

    case.client.get_effect = failure
    monkeypatch.setattr(
        case.module, "cleanup", lambda *args: (_ for _ in ()).throw(RuntimeError("must-not-log-" + case.key))
    )
    with pytest.raises(RuntimeError, match="Exact-instance cleanup") as error:
        case.module.handler(event(case), context(case))
    output = capsys.readouterr().out
    assert case.key not in output + str(error.value)
    assert "must-not-log" not in output + str(error.value)


class ConditionalError(Exception):
    response: ClassVar = {"Error": {"Code": "ConditionalCheckFailedException"}}


class FakeDynamo:
    def __init__(self):
        self.item = None

    def put_item(self, **kwargs):
        assert kwargs["ConditionExpression"] == "attribute_not_exists(lease_id)"
        if self.item:
            raise ConditionalError()
        self.item = copy.deepcopy(kwargs["Item"])

    def get_item(self, **kwargs):
        assert kwargs["ConsistentRead"] is True
        return {"Item": copy.deepcopy(self.item)}

    def update_item(self, **kwargs):
        values = kwargs["ExpressionAttributeValues"]
        if kwargs["UpdateExpression"].startswith("SET lease_owner"):
            if self.item["spec_sha256"] != values[":hash"] or int(self.item.get("owner_until", {"N": "0"})["N"]) >= int(
                values[":now"]["N"]
            ):
                raise ConditionalError()
            self.item.update(lease_owner=values[":owner"], owner_until=values[":until"])
        elif kwargs["UpdateExpression"].startswith("SET state_json"):
            assert "revision = :revision" in kwargs["ConditionExpression"]
            if (
                self.item["lease_owner"] != values[":owner"]
                or self.item["revision"] != values[":revision"]
                or int(self.item["owner_until"]["N"]) < int(values[":now"]["N"])
            ):
                raise ConditionalError()
            self.item.update(state_json=values[":state"], revision=values[":next"])
        else:
            assert kwargs["UpdateExpression"] == "REMOVE lease_owner, owner_until"
            if self.item["lease_owner"] != values[":owner"]:
                raise ConditionalError()
            del self.item["lease_owner"], self.item["owner_until"]
        return {"Attributes": copy.deepcopy(self.item)}


def test_dynamo_owner_lease_blocks_duplicates_and_recovers_after_timeout(module):
    client = FakeDynamo()
    first = module.DynamoState(client, "table", "a" * 32, "hash")
    second = module.DynamoState(client, "table", "a" * 32, "hash")
    state = first.acquire("first", 100, 200)
    state["attempts"] = 1
    first.save(state, 110)
    with pytest.raises(module.LeaseBusyError):
        second.acquire("second", 150, 250)
    recovered = second.acquire("second", 201, 301)
    assert recovered["attempts"] == 1
    with pytest.raises(ConditionalError):
        first.save(state, 202)
    with pytest.raises(ConditionalError):
        first.release()
    second.release()
    assert first.acquire("next", 203, 303)["attempts"] == 1


def test_dynamo_refuses_reusing_lease_id_with_changed_specification(module):
    client = FakeDynamo()
    first = module.DynamoState(client, "table", "a" * 32, "one")
    first.acquire("first", 100, 200)
    first.release()
    with pytest.raises(ValueError, match="specification differs"):
        module.DynamoState(client, "table", "a" * 32, "two").acquire("second", 201, 301)


def test_template_binds_version_retries_secret_and_exact_lease_permissions():
    template = json.loads((ROOT / "scripts/lambda/aws-cleanup.json").read_text())
    resources = template["Resources"]
    function = resources["CleanupFunction"]["Properties"]
    assert function["Handler"] == "aws_cleanup.handler"
    assert function["Timeout"] == 900
    assert "VpcConfig" not in function
    assert "ReservedConcurrentExecutions" not in function
    assert "CleanupApiKey" not in template["Parameters"]
    assert resources["CleanupVersion"]["Properties"]["CodeSha256"] == {"Ref": "CodeSha256"}
    asynchronous = resources["AsyncInvocation"]["Properties"]
    assert asynchronous["Qualifier"] == {"Fn::GetAtt": ["CleanupVersion", "Version"]}
    assert asynchronous["MaximumRetryAttempts"] == 2
    assert asynchronous["MaximumEventAgeInSeconds"] == 3600
    schedule = resources["DeadlineSchedule"]["Properties"]
    assert schedule["ScheduleExpressionTimezone"] == "UTC"
    assert schedule["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert schedule["State"] == "ENABLED"
    assert schedule["Target"]["Arn"] == {"Ref": "CleanupVersion"}
    assert schedule["Target"]["RetryPolicy"]["MaximumEventAgeInSeconds"] == 3600
    statements = resources["CleanupRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    secret = next(value for value in statements if value["Sid"] == "ScopedCleanupCredential")
    assert secret["Action"] == "secretsmanager:GetSecretValue"
    assert secret["Resource"] == {"Ref": "CleanupSecretArn"}
    database = next(value for value in statements if value["Sid"] == "OwnLeaseState")
    assert database["Condition"]["ForAllValues:StringEquals"]["dynamodb:LeadingKeys"] == [{"Ref": "LeaseId"}]
