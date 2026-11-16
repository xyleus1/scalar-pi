"""Deadline cleanup tests use fake clocks and APIs; no cloud resources are touched."""

import copy
import datetime as dt
import importlib.util
import itertools
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def watchdog(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/lambda"))
    spec = importlib.util.spec_from_file_location("scalar_pi_watchdog_test", ROOT / "scripts/lambda/watchdog.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for key, value in {
        "GITHUB_ACTIONS": "true",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "GITHUB_REPOSITORY": "xyleus1/scalar-pi",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_RUN_ID": "12345",
        "GITHUB_RUN_ATTEMPT": "1",
        "WATCHDOG_JOB_STARTED_UNIX": "1800000000",
    }.items():
        monkeypatch.setenv(key, value)
    return module


class Clock:
    def __init__(self):
        self.value = 1800000000

    def now(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.value += seconds


class FakeLambda:
    def __init__(self, module, clock):
        self.module = module
        self.clock = clock
        self.calls = []
        self.instance = None
        self.visible = True
        self.list_errors = []
        self.detail_errors = []
        self.terminate_effect = None
        self.terminate_at = None

    def request(self, method, endpoint, body=None):
        self.calls.append((self.clock.now(), method, endpoint, body))
        if method == "GET" and endpoint == "instances":
            if self.list_errors:
                raise self.list_errors.pop(0)
            return [copy.deepcopy(self.instance)] if self.instance and self.visible else []
        if method == "GET" and endpoint.startswith("instances/"):
            assert endpoint == "instances/" + "b" * 32
            if self.detail_errors:
                raise self.detail_errors.pop(0)
            if self.terminate_at is not None and self.clock.now() >= self.terminate_at:
                return {"id": "b" * 32, "status": "terminated"}
            return copy.deepcopy(self.instance)
        if method == "POST" and endpoint == "instance-operations/terminate":
            assert body == {"instance_ids": ["b" * 32]}
            if self.terminate_effect:
                return self.terminate_effect()
            self.instance["status"] = "terminating"
            self.terminate_at = self.clock.now() + 20
            return {"terminated_instances": [{"id": "b" * 32}]}
        pytest.fail(f"Unexpected API operation: {method} {endpoint}")

    def posts(self):
        return [call for call in self.calls if call[1] == "POST"]


@pytest.fixture
def setup(watchdog, tmp_path):
    clock = Clock()
    request = {
        "instance_name": "scalar-pi-0123456789ab",
        "workspace_id": "c" * 32,
        "worker_commit": watchdog.WORKER_COMMIT,
        "instance_type": "gpu_1x_h100_pcie",
        "region": "us-west-3",
        "deadline_utc": dt.datetime.fromtimestamp(clock.now() + 180, dt.UTC).isoformat(),
        "launch_window_seconds": 60,
    }
    client = FakeLambda(watchdog, clock)
    receipt = watchdog.arm(
        request, tmp_path / "armed.json", execute=True, client=client, now=clock.now, sleep=clock.sleep
    )
    client.instance = {
        "id": "b" * 32,
        "name": request["instance_name"],
        "workspace_id": request["workspace_id"],
        "region": {"name": request["region"]},
        "instance_type": {"name": request["instance_type"]},
        "status": "active",
        "actions": {"terminate": {"available": True}},
        "tags": [{"key": key, "value": value} for key, value in receipt["expected"]["tags"].items()],
    }
    return SimpleNamespace(
        module=watchdog, clock=clock, request=request, client=client, receipt=receipt, output=tmp_path / "result.json"
    )


def run(setup):
    return setup.module.watch(
        setup.receipt, setup.output, execute=True, client=setup.client, now=setup.clock.now, sleep=setup.clock.sleep
    )


def test_exact_target_terminates_only_at_deadline(setup):
    result = run(setup)
    assert result["status"] == "terminated"
    assert result["confirmation"] == "provider_status"
    assert len(setup.client.posts()) == 1
    assert setup.client.posts()[0][0] >= setup.receipt["deadline_unix"]
    assert json.loads(setup.output.with_name("binding.json").read_text())["instance_id"] == "b" * 32


def test_preexisting_name_cannot_arm(setup, tmp_path):
    with pytest.raises(ValueError, match="already exists"):
        setup.module.arm(
            setup.request,
            tmp_path / "another.json",
            execute=True,
            client=setup.client,
            now=setup.clock.now,
            sleep=setup.clock.sleep,
        )
    assert not setup.client.posts()


@pytest.mark.parametrize("field", ["workspace_id", "name", "region", "instance_type", "tags"])
def test_missing_identity_field_never_terminates(setup, field):
    del setup.client.instance[field]
    if field == "name":
        assert run(setup)["status"] == "not_observed"
    else:
        with pytest.raises(ValueError, match="identity|tags"):
            run(setup)
    assert not setup.client.posts()


@pytest.mark.parametrize("key", ["project", "commit", "watchdog-run", "watchdog-deadline"])
def test_wrong_identity_tag_never_terminates(setup, key):
    for tag in setup.client.instance["tags"]:
        if tag["key"] == key:
            tag["value"] = "wrong"
    with pytest.raises(ValueError, match="tags"):
        run(setup)
    assert not setup.client.posts()


def test_duplicate_names_never_choose_a_candidate(setup, monkeypatch):
    original = setup.client.request

    def request(method, endpoint, body=None):
        if endpoint == "instances":
            return [setup.client.instance, {**setup.client.instance, "id": "d" * 32}]
        return original(method, endpoint, body)

    monkeypatch.setattr(setup.client, "request", request)
    with pytest.raises(ValueError, match="Duplicate"):
        run(setup)
    assert not setup.client.posts()


def test_identity_change_after_binding_never_terminates(setup, monkeypatch):
    original = setup.client.request

    def request(method, endpoint, body=None):
        value = original(method, endpoint, body)
        if endpoint.startswith("instances/"):
            value["workspace_id"] = "d" * 32
        return value

    monkeypatch.setattr(setup.client, "request", request)
    with pytest.raises(ValueError, match="identity"):
        run(setup)
    assert not setup.client.posts()


def test_bound_id_404_is_terminal(setup):
    setup.client.detail_errors.append(setup.module.LambdaApiError("GET", "instances/id", 404))
    result = run(setup)
    assert result["confirmation"] == "bound_id_404"
    assert not setup.client.posts()


def test_forbidden_read_is_not_a_missing_instance(setup):
    setup.client.detail_errors.append(setup.module.LambdaApiError("GET", "instances/id", 403))
    with pytest.raises(setup.module.LambdaApiError):
        run(setup)
    assert json.loads(setup.output.read_text())["status"] == "failed"
    assert not setup.client.posts()


def test_unavailable_termination_is_read_only(setup):
    setup.client.instance["actions"]["terminate"]["available"] = False
    result = run(setup)
    assert result["status"] == "unresolved"
    assert not setup.client.posts()


def test_already_terminating_never_posts(setup):
    setup.client.instance["status"] = "terminating"
    setup.client.terminate_at = setup.clock.now() + 20
    assert run(setup)["status"] == "terminated"
    assert not setup.client.posts()


def test_minimal_terminating_response_keeps_polling_bound_id(setup, monkeypatch):
    original = setup.client.request
    statuses = iter(("terminating", "terminated"))

    def request(method, endpoint, body=None):
        if endpoint.startswith("instances/"):
            return {"id": "b" * 32, "status": next(statuses)}
        return original(method, endpoint, body)

    monkeypatch.setattr(setup.client, "request", request)
    assert run(setup)["status"] == "terminated"
    assert not setup.client.posts()


def test_acknowledged_post_is_never_repeated_even_if_active_status_lags(setup):
    setup.client.terminate_effect = lambda: {"terminated_instances": [{"id": "b" * 32}]}
    result = run(setup)
    assert result["status"] == "unresolved"
    assert result["termination_acknowledged"] is True
    assert len(setup.client.posts()) == 1


def test_uncertain_post_then_terminating_is_not_repeated(setup):
    def terminate():
        setup.client.instance["status"] = "terminating"
        setup.client.terminate_at = setup.clock.now() + 30
        raise TimeoutError("Response lost")

    setup.client.terminate_effect = terminate
    assert run(setup)["status"] == "terminated"
    assert len(setup.client.posts()) == 1


def test_uncertain_post_requires_reconciliation_before_bounded_retry(setup):
    def terminate():
        raise TimeoutError("No acknowledgement")

    setup.client.terminate_effect = terminate
    assert run(setup)["status"] == "unresolved"
    calls = setup.client.posts()
    assert len(calls) == setup.module.MAX_TERMINATE_ATTEMPTS
    assert all(second[0] - first[0] >= 60 for first, second in itertools.pairwise(calls))
    for first, second in itertools.pairwise(calls):
        reads = [call for call in setup.client.calls if first[0] < call[0] <= second[0] and call[1] == "GET"]
        assert len(reads) >= 3


def test_forbidden_post_is_never_retried(setup):
    def terminate():
        raise setup.module.LambdaApiError("POST", "instance-operations/terminate", 403)

    setup.client.terminate_effect = terminate
    with pytest.raises(setup.module.LambdaApiError):
        run(setup)
    assert len(setup.client.posts()) == 1


def test_transient_reads_do_not_move_absolute_deadline(setup):
    setup.client.list_errors = [setup.module.LambdaApiError("GET", "instances", 429), TimeoutError()]
    setup.client.detail_errors = [setup.module.LambdaApiError("GET", "instances/id", 503)]
    assert run(setup)["status"] == "terminated"
    assert setup.client.posts()[0][0] == setup.receipt["deadline_unix"]


def test_late_visibility_is_still_covered_after_admission_closes(setup, monkeypatch):
    original = setup.client.request

    def request(method, endpoint, body=None):
        setup.client.visible = setup.clock.now() > setup.receipt["deadline_unix"]
        return original(method, endpoint, body)

    monkeypatch.setattr(setup.client, "request", request)
    assert run(setup)["status"] == "terminated"
    assert setup.client.posts()[0][0] > setup.receipt["deadline_unix"]


@pytest.mark.parametrize("field", ["deadline_unix", "launch_not_after_unix", "cleanup_deadline_unix"])
def test_changed_deadlines_are_rejected_before_api_access(setup, field):
    setup.receipt[field] += 1
    before = len(setup.client.calls)
    with pytest.raises(ValueError, match="deadline|window"):
        run(setup)
    assert len(setup.client.calls) == before


def test_expired_watch_start_refuses_admission(setup):
    setup.clock.value = setup.receipt["launch_not_after_unix"]
    before = len(setup.client.calls)
    with pytest.raises(TimeoutError):
        run(setup)
    assert len(setup.client.calls) == before


def test_slow_setup_cannot_arm_beyond_job_timeout(setup, monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHDOG_JOB_STARTED_UNIX", str(setup.clock.now() - 5 * 3600))
    before = len(setup.client.calls)
    with pytest.raises(ValueError, match="job time limit"):
        setup.module.arm(
            setup.request,
            tmp_path / "too-late.json",
            execute=True,
            client=setup.client,
            now=setup.clock.now,
            sleep=setup.clock.sleep,
        )
    assert len(setup.client.calls) == before


def test_request_cannot_exceed_four_hour_lease(setup, tmp_path):
    setup.request["deadline_utc"] = dt.datetime.fromtimestamp(setup.clock.now() + 4 * 3600 + 1, dt.UTC).isoformat()
    before = len(setup.client.calls)
    with pytest.raises(ValueError, match="four hours"):
        setup.module.arm(
            setup.request,
            tmp_path / "too-long.json",
            execute=True,
            client=setup.client,
            now=setup.clock.now,
            sleep=setup.clock.sleep,
        )
    assert len(setup.client.calls) == before


def test_transient_read_failure_stops_at_cleanup_deadline(setup, monkeypatch):
    def request(method, endpoint, body=None):
        assert method == "GET"
        raise TimeoutError("Connection lost")

    monkeypatch.setattr(setup.client, "request", request)
    with pytest.raises(TimeoutError, match="absolute cleanup deadline"):
        run(setup)
    assert setup.clock.now() == setup.receipt["cleanup_deadline_unix"]
    assert json.loads(setup.output.read_text())["status"] == "failed"


def test_slow_final_read_cannot_post_after_cleanup_deadline(setup, monkeypatch):
    original = setup.client.request

    def request(method, endpoint, body=None):
        result = original(method, endpoint, body)
        if endpoint.startswith("instances/"):
            setup.clock.value = setup.receipt["cleanup_deadline_unix"] + 1
        return result

    monkeypatch.setattr(setup.client, "request", request)
    assert run(setup)["status"] == "unresolved"
    assert not setup.client.posts()


def test_terminal_response_cannot_change_bound_id(setup, monkeypatch):
    original = setup.client.request

    def request(method, endpoint, body=None):
        if endpoint.startswith("instances/"):
            return {"id": "d" * 32, "status": "terminated"}
        return original(method, endpoint, body)

    monkeypatch.setattr(setup.client, "request", request)
    with pytest.raises(ValueError, match="Bound instance ID changed"):
        run(setup)
    assert not setup.client.posts()


def test_dry_run_constructs_no_client_and_cannot_authorize_live_watch(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(setup.module, "LambdaClient", lambda *_: pytest.fail("Dry run loaded a cloud credential"))
    receipt = setup.module.arm(setup.request, tmp_path / "dry-armed.json", now=setup.clock.now)
    assert receipt["dry_run"] is True
    result = setup.module.watch(
        receipt, tmp_path / "dry-result.json", now=setup.clock.now, sleep=setup.clock.sleep, dry_run_seconds=10
    )
    assert result["api_requests"] == 0
    with pytest.raises(ValueError, match="interchanged"):
        setup.module.watch(receipt, tmp_path / "live-result.json", execute=True, now=setup.clock.now)


def test_workflow_uses_pinned_actions_and_no_automatic_trigger():
    document = yaml.safe_load((ROOT / ".github/workflows/lambda-watchdog.yml").read_text())
    assert set(document.get("on", document.get(True))) == {"workflow_dispatch"}
    assert document["permissions"] == {"contents": "read"}
    assert document["concurrency"]["cancel-in-progress"] is False
    job = document["jobs"]["cleanup"]
    assert job["timeout-minutes"] < 360
    assert "refs/heads/main" in job["if"]
    steps = job["steps"]
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"actions/[a-z-]+@[a-f0-9]{40}", step["uses"])
    assert next(step for step in steps if "checkout@" in step.get("uses", ""))["with"]["persist-credentials"] is False
    names = [step["name"] for step in steps]
    assert names.index("Arm cleanup") < names.index("Publish armed receipt") < names.index("Watch deadline")
    assert job["environment"] == "lambda-cleanup"
