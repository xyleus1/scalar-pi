"""Launch admission tests use fake GitHub/Lambda APIs and never create resources."""

import base64
import copy
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
NOW = 1_800_000_000
KEY = "fixture-expiring-creator-key"
MASTER = "fixture-master-read-only-key"
INSTANCE_ID = "b" * 32


@pytest.fixture
def guarded(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/lambda"))
    spec = importlib.util.spec_from_file_location("scalar_pi_guarded_test", ROOT / "scripts/lambda/guarded.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(guarded, tmp_path, monkeypatch):
    config = {
        "region": "us-west-3",
        "instance_type": "gpu_1x_h100_pcie",
        "image_id": "a" * 32,
        "operator_cidr": "203.0.113.1/32",
        "max_hourly_rate_usd": 3.79,
        "budget_usd": 2000,
        "qualification_budget_usd": 100,
        "noncompute_spent_usd": 0,
        "noncompute_reserve_usd": 100,
        "qualification_noncompute_reserve_usd": 20,
        "credits_verified": True,
    }
    request = {
        "instance_name": "scalar-pi-0123456789ab",
        "workspace_id": "c" * 32,
        "worker_commit": guarded.WORKER_COMMIT,
        "region": config["region"],
        "instance_type": config["instance_type"],
        "deadline_utc": dt.datetime.fromtimestamp(NOW + 3600, dt.UTC).isoformat(),
        "launch_window_seconds": 600,
    }
    expected = {
        key: request[key] for key in ("instance_name", "workspace_id", "worker_commit", "region", "instance_type")
    }
    expected["tags"] = {
        "project": "scalar-pi",
        "commit": guarded.WORKER_COMMIT,
        "watchdog-run": "1234-1",
        "watchdog-deadline": str(NOW + 3600),
    }
    armed = {
        "schema_version": 1,
        "kind": "lambda-deadline-watchdog",
        "status": "armed",
        "dry_run": False,
        "armed_unix": NOW - 1,
        "launch_not_after_unix": NOW + 599,
        "deadline_unix": NOW + 3600,
        "request": request,
        "request_sha256": guarded.sha256(request),
        "expected": expected,
        "github": {"run_id": "1234", "run_attempt": "1", "head_sha": "d" * 40},
    }
    receipt = {
        "schema_version": 1,
        "identity_id": "identity-fixture",
        "key_id": "key-fixture",
        "workspace_id": request["workspace_id"],
        "workspace_ids": [request["workspace_id"]],
        "role_ids": [guarded.LAUNCH_ROLE_ID],
        "role_names": ["workspace-member"],
        "account_role_ids": [],
        "inherited_role_ids": [],
        "group_ids": [],
        "permission_ids": ["compute:instance:launch", "compute:instance:read", "compute:instance:terminate"],
        "expires_at": dt.datetime.fromtimestamp(NOW + 300, dt.UTC).isoformat(),
        "key_sha256": hashlib.sha256(KEY.encode()).hexdigest(),
    }
    script = tmp_path / "worker.sh"
    script.write_text("printf 'qualification fixture\\n'\n")
    receipt_path = tmp_path / "creator-key.json"
    receipt_path.write_text(json.dumps(receipt))
    state_path = tmp_path / "lambda-state.json"
    state = {"schema_version": 1, "resources": {"ssh_key_id": "fixture-key", "filesystem_id": "fixture-fs"}, "runs": []}
    guarded.cloud.save_state(state_path, state)
    calls = []

    class Reader:
        key = MASTER

        def request(self, method, endpoint, body=None):
            calls.append(("master", method, endpoint, body))
            assert method == "GET"
            if endpoint == "instance-types":
                return {
                    config["instance_type"]: {
                        "instance_type": {"price_cents_per_hour": 379, "specs": {"gpus": 1}, "architecture": "x86_64"},
                        "regions_with_capacity_available": [{"name": config["region"]}],
                    }
                }
            if endpoint == "images":
                return [
                    {
                        "id": config["image_id"],
                        "architecture": "x86_64",
                        "family": "gpu-base-22-04",
                        "region": {"name": config["region"]},
                    }
                ]
            if endpoint == "instances":
                return []
            pytest.fail(f"Unexpected read endpoint {endpoint}")

    class Creator:
        def request(self, method, endpoint, body=None):
            calls.append(("creator", method, endpoint, body))
            assert method == "POST"
            assert endpoint == "instance-operations/launch"
            assert guarded.cloud.load_state(state_path)["pending_launch"]["name"] == request["instance_name"]
            return {"instance_ids": [INSTANCE_ID]}

    creator = Creator()

    def make_creator(key):
        assert key == KEY
        return creator

    monkeypatch.setattr(guarded.cloud, "LambdaClient", make_creator)

    class Guard:
        def wait_ready(self, path, *, timeout_seconds):
            calls.append(("guard", "ready", None, None))
            return copy.deepcopy(armed)

        def revalidate(self, path, *, expected_request):
            assert expected_request == request
            calls.append(("guard", "revalidate", None, None))
            return copy.deepcopy(armed)

    guard = Guard()
    kwargs = {
        "config": config,
        "state_path": state_path,
        "script_path": script,
        "key_receipt_path": receipt_path,
        "guard_path": tmp_path / "guard.json",
        "request": request,
        "guard": guard,
        "reader": Reader(),
        "launch_key": KEY,
        "now": lambda: NOW,
    }
    return SimpleNamespace(
        config=config,
        request=request,
        armed=armed,
        receipt=receipt,
        receipt_path=receipt_path,
        state_path=state_path,
        calls=calls,
        creator=creator,
        guard=guard,
        kwargs=kwargs,
    )


def test_launch_is_once_detached_bound_and_recorded(guarded, setup):
    run = guarded.launch(**setup.kwargs)
    mutations = [call for call in setup.calls if call[1] == "POST"]
    assert len(mutations) == 1
    assert mutations[0][0] == "creator"
    assert setup.calls[-2][0:2] == ("guard", "revalidate")
    body = mutations[0][3]
    assert body["workspace_id"] == setup.request["workspace_id"]
    assert {item["key"]: item["value"] for item in body["tags"]} == setup.armed["expected"]["tags"]
    assert len(body["user_data"].encode()) <= guarded.USER_DATA_LIMIT
    cloud_init = yaml.safe_load(body["user_data"])
    assert "nvidia-ctk" in cloud_init["runcmd"][-2][2]
    bootstrap = cloud_init["runcmd"][-1][2]
    assert " monitor " not in bootstrap
    payloads = re.findall(r"printf '%s' '([A-Za-z0-9+/=]+)'", bootstrap)
    worker = base64.b64decode(payloads[1]).decode()
    assert f"git checkout --detach {guarded.WORKER_COMMIT}" in worker
    assert f"SCALAR_PI_DEADLINE_UNIX={NOW + 3600}\n" in worker
    assert MASTER not in worker
    assert KEY not in worker
    assert run["instance_id"] == INSTANCE_ID
    assert run["deadline_unix"] == NOW + 3600
    assert run["max_hours"] == (run["deadline_unix"] - run["started_unix"]) / 3600
    assert run["creator_key"]["key_sha256"] == setup.receipt["key_sha256"]
    state = guarded.cloud.load_state(setup.state_path)
    assert "pending_launch" not in state
    assert state["runs"] == [run]
    assert MASTER not in setup.state_path.read_text()
    assert KEY not in setup.state_path.read_text()


@pytest.mark.parametrize("offset", [-1, 60, 600, 3600])
def test_creator_key_expiry_blocks_launch(guarded, setup, offset):
    setup.receipt["expires_at"] = dt.datetime.fromtimestamp(NOW + offset, dt.UTC).isoformat()
    setup.receipt_path.write_text(json.dumps(setup.receipt))
    with pytest.raises(ValueError, match="Creator key must expire"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


@pytest.mark.parametrize(
    "change",
    [
        {"key_sha256": "f" * 64},
        {"workspace_id": "f" * 32},
        {"permission_ids": ["compute:instance:terminate"]},
        {"permission_ids": ["compute:instance:create", "compute:instance:terminate"]},
        {"permission_ids": ["compute:instance:launch", "compute:instance:launch"]},
        {"role_ids": ["f" * 32]},
        {"role_names": ["account-admin"]},
        {"workspace_ids": ["f" * 32]},
        {"account_role_ids": ["extra-role"]},
        {"inherited_role_ids": ["extra-role"]},
        {"group_ids": ["extra-group"]},
        {"key_id": ""},
        {"expires_at": "2027-01-15T08:00:00"},
    ],
)
def test_invalid_creator_receipt_blocks_launch(guarded, setup, change):
    setup.receipt.update(change)
    setup.receipt_path.write_text(json.dumps(setup.receipt))
    with pytest.raises(ValueError, match="Creator"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_master_key_cannot_launch(guarded, setup):
    setup.kwargs["launch_key"] = MASTER
    with pytest.raises(ValueError, match="separate"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


@pytest.mark.parametrize("failure", [RuntimeError("Watch step stopped"), TimeoutError("Guard expired")])
def test_failed_live_guard_blocks_launch(guarded, setup, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(setup.guard, "revalidate", fail)
    with pytest.raises(type(failure)):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)
    assert "pending_launch" not in guarded.cloud.load_state(setup.state_path)


def test_suspend_after_pending_prevents_post(guarded, setup, monkeypatch):
    clock = [NOW]
    setup.kwargs["now"] = lambda: clock[0]
    save = guarded.cloud.save_state

    def suspend_after_save(path, state):
        save(path, state)
        if state.get("pending_launch"):
            clock[0] += 7200

    monkeypatch.setattr(guarded.cloud, "save_state", suspend_after_save)
    with pytest.raises(ValueError, match="Creator key must expire"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)
    assert "pending_launch" not in guarded.cloud.load_state(setup.state_path)


@pytest.mark.parametrize("status", [400, 401, 403, 409, 422, 429])
def test_definitive_rejection_clears_pending(guarded, setup, monkeypatch, status):
    def reject(method, endpoint, body):
        setup.calls.append(("creator", method, endpoint, body))
        raise guarded.cloud.LambdaApiError(method, endpoint, status)

    monkeypatch.setattr(setup.creator, "request", reject)
    with pytest.raises(guarded.cloud.LambdaApiError):
        guarded.launch(**setup.kwargs)
    assert sum(call[1] == "POST" for call in setup.calls) == 1
    assert "pending_launch" not in guarded.cloud.load_state(setup.state_path)


@pytest.mark.parametrize("failure", ["transport", "interrupt", "408", "499", "500", "malformed", "multiple"])
def test_ambiguous_response_keeps_pending_and_never_retries(guarded, setup, monkeypatch, failure):
    def uncertain(method, endpoint, body):
        setup.calls.append(("creator", method, endpoint, body))
        if failure == "transport":
            raise RuntimeError("connection lost")
        if failure == "interrupt":
            raise KeyboardInterrupt()
        if failure.isdigit():
            raise guarded.cloud.LambdaApiError(method, endpoint, int(failure))
        return {"instance_ids": [INSTANCE_ID, INSTANCE_ID]} if failure == "multiple" else None

    monkeypatch.setattr(setup.creator, "request", uncertain)
    with pytest.raises(RuntimeError, match="uncertain"):
        guarded.launch(**setup.kwargs)
    state = guarded.cloud.load_state(setup.state_path)
    assert state["pending_launch"]["name"] == setup.request["instance_name"]
    assert state["runs"] == []
    with pytest.raises(ValueError, match="still open"):
        guarded.launch(**setup.kwargs)
    assert sum(call[1] == "POST" for call in setup.calls) == 1


def test_changed_guard_and_dry_receipt_cannot_authorize(guarded, setup, monkeypatch):
    setup.armed["dry_run"] = True
    with pytest.raises(ValueError, match="Live watchdog"):
        guarded.launch(**setup.kwargs)
    setup.armed["dry_run"] = False

    def changed(*args, **kwargs):
        return {**setup.armed, "deadline_unix": NOW + 7200}

    monkeypatch.setattr(setup.guard, "revalidate", changed)
    with pytest.raises(ValueError, match="changed"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_payload_size_limit_blocks_before_post(guarded, setup):
    setup.kwargs["script_path"].write_text("#" + "x" * guarded.USER_DATA_LIMIT)
    with pytest.raises(ValueError, match="one-million-byte"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_budget_and_source_binding_remain_enforced(guarded, setup):
    setup.config["qualification_budget_usd"] = 1
    with pytest.raises(ValueError, match="phase allowance"):
        guarded.launch(**setup.kwargs)
    setup.config["qualification_budget_usd"] = 100
    report_path = setup.state_path.with_name("budget.json")
    report_path.write_text(json.dumps({"experiment_identity": {"code_commit": "a" * 40}}))
    setup.kwargs["budget_report_path"] = report_path
    with pytest.raises(ValueError, match="different scientific source"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_research_reuses_the_measured_study_binding(guarded, setup):
    report = {
        "selected_study": "pilot",
        "budget_usd": 2000,
        "live_hourly_rate_usd": 3.79,
        "forecast_hourly_rate_usd": 3.79,
        "remaining_compute_cost_with_headroom_usd": 1000,
        "remaining_noncompute_cost_usd": 100,
        "experiment_identity": {"code_commit": guarded.WORKER_COMMIT, "container_image_id": "sha256:" + "a" * 64},
        "study_lock_sha256": "e" * 64,
    }
    report_path = setup.state_path.with_name("budget.json")
    report_path.write_text(json.dumps({"budget": report}))
    setup.kwargs.update(stage="research", budget_report_path=report_path)
    run = guarded.launch(**setup.kwargs)
    state = guarded.cloud.load_state(setup.state_path)
    assert state["study_decision"]["experiment_identity"] == report["experiment_identity"]
    assert state["study_decision"]["study_lock_sha256"] == report["study_lock_sha256"]
    assert run["budget_report_sha256"] == hashlib.sha256(report_path.read_bytes()).hexdigest()


def test_elapsed_preflight_shortens_original_lease(guarded, setup, monkeypatch):
    clock = [NOW]
    setup.kwargs["now"] = lambda: clock[0]

    def slower_live_check(*args, **kwargs):
        clock[0] += 15.125
        return copy.deepcopy(setup.armed)

    monkeypatch.setattr(setup.guard, "revalidate", slower_live_check)
    run = guarded.launch(**setup.kwargs)
    assert run["deadline_unix"] == NOW + 3600
    assert run["started_unix"] == NOW + 15.125
    assert run["max_hours"] == (3600 - 15.125) / 3600


@pytest.mark.parametrize("secret", [KEY, MASTER])
def test_worker_cannot_embed_api_credentials(guarded, setup, secret):
    setup.kwargs["script_path"].write_text("export CREDENTIAL=" + secret)
    with pytest.raises(ValueError, match="credentials"):
        guarded.launch(**setup.kwargs)
    assert setup.calls == []


def test_existing_project_instance_blocks_launch(guarded, setup, monkeypatch):
    reader = setup.kwargs["reader"]
    original = reader.request

    def existing(method, endpoint, body=None):
        if endpoint == "instances":
            return [
                {"name": "scalar-pi-111111111111", "workspace_id": setup.request["workspace_id"], "status": "active"}
            ]
        return original(method, endpoint, body)

    monkeypatch.setattr(reader, "request", existing)
    with pytest.raises(ValueError, match="existing project instance"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_closed_attempt_cannot_be_launched_again(guarded, setup):
    state = guarded.cloud.load_state(setup.state_path)
    state["runs"].append({"name": setup.request["instance_name"], "ended_unix": NOW - 1})
    guarded.cloud.save_state(setup.state_path, state)
    with pytest.raises(ValueError, match="cannot be reused"):
        guarded.launch(**setup.kwargs)
    assert setup.calls == []


def test_dry_run_does_not_authenticate_or_create_state(guarded, setup, tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    request_path = tmp_path / "request.json"
    config_path.write_text(json.dumps(setup.config))
    request_path.write_text(json.dumps(setup.request))
    monkeypatch.setattr(guarded.cloud, "LambdaClient", lambda *_: pytest.fail("Dry run authenticated"))
    assert (
        guarded.main(
            [
                "--config",
                str(config_path),
                "--request",
                str(request_path),
                "--guard",
                "missing.json",
                "--key-receipt",
                "missing-key.json",
                "--script",
                "missing.sh",
                "--controller-commit",
                "d" * 40,
                "--gh",
                "gh.exe",
                "--evidence-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["resources_created"] == 0
