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


def portable_setup(guarded, setup, monkeypatch, region="us-southeast-1"):
    """Keep real admission/serialization; change only provider fixture facts."""
    setup.config.update(storage_mode="replicated_local", instance_type="gpu_2x_h100_sxm5",
                        region=region, max_hourly_rate_usd=9.58)
    setup.request.update(storage_mode="replicated_local", instance_type="gpu_2x_h100_sxm5", region=region)
    setup.armed["expected"].update(instance_type="gpu_2x_h100_sxm5", region=region)
    setup.armed["request_sha256"] = guarded.sha256(setup.request)
    setup.kwargs.update(existing_checkout=False, normalization_sha256=None)
    state = guarded.cloud.load_state(setup.state_path)
    del state["resources"]["filesystem_id"]
    guarded.cloud.save_state(setup.state_path, state)
    reader = setup.kwargs["reader"]
    original = reader.request

    def read(method, endpoint, body=None):
        if endpoint == "instance-types":
            setup.calls.append(("master", method, endpoint, body))
            return {setup.config["instance_type"]: {
                "instance_type": {"price_cents_per_hour": 958, "specs": {"gpus": 2}, "architecture": "x86_64"},
                "regions_with_capacity_available": [{"name": region}],
            }}
        return original(method, endpoint, body)

    monkeypatch.setattr(reader, "request", read)


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


@pytest.mark.parametrize("region", ["us-southeast-1", "us-south-3"])
def test_portable_launch_has_true_local_mount_before_unchanged_detached_worker(guarded, setup, monkeypatch, region):
    portable_setup(guarded, setup, monkeypatch, region)
    run = guarded.launch(**setup.kwargs)
    mutations = [call for call in setup.calls if call[1] == "POST"]
    assert len(mutations) == 1
    body = mutations[0][3]
    assert body["region_name"] == region and body["instance_type_name"] == "gpu_2x_h100_sxm5"
    assert "file_system_mounts" not in body
    commands = yaml.safe_load(body["user_data"])["runcmd"]
    storage, detached_command = commands[-2:]
    assert storage[:2] == ["python3", "-c"]
    assert "replicated_local" in storage[2] and "ephemeral" in storage[2]
    assert storage[storage.index("--attempt") + 1] == setup.request["instance_name"]
    assert float(storage[storage.index("--deadline-unix") + 1]) == NOW + 3600
    worker = base64.b64decode(re.findall(r"printf '%s' '([A-Za-z0-9+/=]+)'", detached_command[2])[1]).decode()
    assert "git clone" in worker and f"git checkout --detach {guarded.WORKER_COMMIT}" in worker
    assert "SCALAR_PI_VERIFY_NORMS" not in worker
    assert MASTER not in body["user_data"] and KEY not in body["user_data"]
    assert run["filesystem_id"] is None and run["filesystem_mount_point"] is None
    assert run["storage_mode"] == "replicated_local" and run["storage_persistence"] == "ephemeral"
    assert run["local_storage_source"] == "/var/lib/scalar-pi-replica"
    assert run["commit"] == guarded.WORKER_COMMIT  # Actual bootstrap checkout, not the scientific image.


@pytest.mark.parametrize("change", ["no_opt_in", "other_region", "one_gpu", "shared_request", "other_request_region",
                                    "research", "existing_checkout"])
def test_portable_mismatches_fail_before_launch(guarded, setup, monkeypatch, change):
    portable_setup(guarded, setup, monkeypatch)
    if change == "no_opt_in":
        del setup.config["storage_mode"]
    elif change == "other_region":
        setup.config["region"] = setup.request["region"] = "us-west-1"
    elif change == "one_gpu":
        setup.config["instance_type"] = setup.request["instance_type"] = "gpu_1x_h100_pcie"
    elif change == "shared_request":
        del setup.request["storage_mode"]
    elif change == "other_request_region":
        setup.request["region"] = "us-south-3"
    elif change == "research":
        setup.kwargs["stage"] = "research"
    else:
        setup.kwargs["existing_checkout"] = True
    with pytest.raises(ValueError):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_portable_watchdog_requires_explicit_request_opt_in(guarded, setup, monkeypatch):
    portable_setup(guarded, setup, monkeypatch)
    assert guarded.validate_request(setup.request, NOW)[0] == setup.request
    missing = {key: value for key, value in setup.request.items() if key != "storage_mode"}
    with pytest.raises(ValueError):
        guarded.validate_request(missing, NOW)
    with pytest.raises(ValueError):
        guarded.validate_request({**setup.request, "region": "us-west-3"}, NOW)


def test_portable_configuration_never_prepares_persistent_storage(guarded, setup, monkeypatch):
    portable_setup(guarded, setup, monkeypatch)
    with pytest.raises(ValueError, match="does not prepare a filesystem"):
        guarded.cloud.prepare(setup.kwargs["reader"], setup.config, {}, setup.state_path)
    with pytest.raises(ValueError, match="pre-armed"):
        guarded.cloud.run_session(setup.kwargs["reader"], setup.config, {}, setup.state_path, SimpleNamespace())
    assert not setup.calls


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


@pytest.fixture
def parallel(guarded, setup, monkeypatch):
    old_id = "e" * 32
    expected = copy.deepcopy(setup.armed["expected"])
    expected["instance_name"] = "scalar-pi-111111111111"
    expected["tags"]["watchdog-run"] = "5678-1"
    lease = {
        "kind": "aws-lambda-scheduler",
        "instance_id": old_id,
        "lease_id": "f" * 32,
        "lease_sha256": "a" * 64,
        "function_version_arn": "arn:aws:lambda:us-east-1:123456789012:function:cleanup:1",
        "schedule_arn": "arn:aws:scheduler:us-east-1:123456789012:schedule/group/cleanup",
        "deadline_unix": NOW + 86400,
        "cleanup_deadline_unix": NOW + 93600,
        "key_expires_unix": NOW + 97200,
        "github_conclusion": "cancelled",
    }
    run = {
        "instance_id": old_id,
        "name": expected["instance_name"],
        "commit": guarded.WORKER_COMMIT,
        "instance_type": setup.config["instance_type"],
        "region": setup.config["region"],
        "workspace_id": setup.request["workspace_id"],
        "started_unix": NOW - 3600,
        "deadline_unix": NOW + 3600,
        "max_hours": 2,
        "hourly_rate_usd": 3.79,
        "stage": "qualification",
        "allowance": {"original": "unchanged"},
        "watchdog": {"expected": expected},
        "cleanup_leases": [lease],
        "effective_cleanup_lease_id": lease["lease_id"],
    }
    state = guarded.cloud.load_state(setup.state_path)
    state["runs"].append(run)
    guarded.cloud.save_state(setup.state_path, state)
    instance = {
        "id": old_id,
        "name": expected["instance_name"],
        "workspace_id": expected["workspace_id"],
        "region": {"name": expected["region"]},
        "instance_type": {"name": expected["instance_type"]},
        "tags": [{"key": key, "value": value} for key, value in expected["tags"].items()],
        "status": "active",
    }
    instances = [instance]
    reader = setup.kwargs["reader"]
    original = reader.request

    def read(method, endpoint, body=None):
        if endpoint == "instances":
            setup.calls.append(("master", method, endpoint, body))
            return copy.deepcopy(instances)
        return original(method, endpoint, body)

    monkeypatch.setattr(reader, "request", read)
    proof = {
        key: lease[key]
        for key in (
            "instance_id",
            "lease_id",
            "lease_sha256",
            "function_version_arn",
            "schedule_arn",
            "deadline_unix",
            "cleanup_deadline_unix",
            "key_expires_unix",
        )
    }
    proof["run_sha256"] = guarded.sha256(run)
    receipt = {"schema_version": 1, "observed_unix": NOW, "runs": [proof]}
    verifications = []

    def verify(opened):
        assert opened == [run]
        opened[0]["allowance"]["original"] = "callback cannot mutate ledger"
        verifications.append(1)
        return copy.deepcopy(receipt)

    setup.config["qualification_budget_usd"] = 500
    setup.kwargs.update(
        protected_active_instance_ids=(old_id,),
        verify_active_leases=verify,
        existing_checkout=True,
        normalization_sha256="c" * 64,
    )
    return SimpleNamespace(run=run, instances=instances, receipt=receipt, verifications=verifications)


def test_parallel_launch_keeps_existing_run_and_reserves_full_cleanup_window(guarded, setup, parallel):
    new = guarded.launch(**setup.kwargs)
    state = guarded.cloud.load_state(setup.state_path)
    assert state["runs"][0] == parallel.run
    assert state["runs"][1] == new
    assert len(parallel.verifications) == 2
    assert new["allowance"]["open_compute_reserve_usd"] == pytest.approx(26 * 3.79 * 1.2)
    assert new["allowance"]["open_qualification_reserve_usd"] == new["allowance"]["open_compute_reserve_usd"]
    assert new["parallel_admission"]["protected_cleanup"] == parallel.receipt
    assert sum(call[1] == "POST" for call in setup.calls) == 1
    body = next(call[3] for call in setup.calls if call[1] == "POST")
    bootstrap = yaml.safe_load(body["user_data"])["runcmd"][-1][2]
    worker = base64.b64decode(re.findall(r"printf '%s' '([A-Za-z0-9+/=]+)'", bootstrap)[1]).decode()
    assert "git fetch" not in worker
    assert "git checkout" not in worker
    assert "git clone" not in worker
    assert "SCALAR_PI_VERIFY_NORMS" in worker
    assert "c" * 64 in worker


def test_portable_launch_protects_existing_west3_host_against_its_own_identity(guarded, setup, parallel, monkeypatch):
    original = copy.deepcopy(parallel.run)
    portable_setup(guarded, setup, monkeypatch)
    new = guarded.launch(**setup.kwargs)
    state = guarded.cloud.load_state(setup.state_path)
    assert state["runs"][0] == original
    assert state["runs"][1] == new
    assert len(parallel.verifications) == 2
    assert new["parallel_admission"]["protected_cleanup"] == parallel.receipt
    assert new["allowance"]["open_compute_reserve_usd"] >= 26 * 3.79 * 1.2
    assert sum(call[1] == "POST" for call in setup.calls) == 1


@pytest.mark.parametrize("change", ["provider_type", "provider_region", "provider_commit", "workspace", "bad_proof",
                                    "unrecorded_instance", "stale_proof"])
def test_portable_protected_host_mismatch_or_stale_proof_still_blocks(guarded, setup, parallel, monkeypatch, change):
    portable_setup(guarded, setup, monkeypatch)
    instance = parallel.instances[0]
    if change == "provider_type":
        instance["instance_type"]["name"] = "gpu_2x_h100_sxm5"
    elif change == "provider_region":
        instance["region"]["name"] = "us-south-3"
    elif change == "provider_commit":
        next(t for t in instance["tags"] if t["key"] == "commit")["value"] = "9" * 40
    elif change == "workspace":
        instance["workspace_id"] = "9" * 32
    elif change == "bad_proof":
        parallel.receipt["runs"][0]["run_sha256"] = "0" * 64
    elif change == "stale_proof":
        parallel.receipt["observed_unix"] = NOW - 31
    else:
        parallel.instances.append({**instance, "id": "9" * 32, "name": "scalar-pi-999999999999"})
    with pytest.raises(ValueError):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_portable_uncertain_post_is_retained_without_retry(guarded, setup, monkeypatch):
    portable_setup(guarded, setup, monkeypatch)
    def uncertain(method, endpoint, body=None):
        setup.calls.append(("creator", method, endpoint, body))
        raise TimeoutError("Fixture connection lost after POST")
    monkeypatch.setattr(setup.creator, "request", uncertain)
    with pytest.raises(RuntimeError, match="Launch response uncertain"):
        guarded.launch(**setup.kwargs)
    state = guarded.cloud.load_state(setup.state_path)
    assert state["pending_launch"]["storage_persistence"] == "ephemeral"
    assert state["pending_launch"]["filesystem_id"] is None
    with pytest.raises(ValueError, match="still open"):
        guarded.launch(**setup.kwargs)
    assert sum(call[1] == "POST" for call in setup.calls) == 1


def test_default_single_instance_gate_is_unchanged(guarded, setup, parallel):
    setup.kwargs.pop("protected_active_instance_ids")
    with pytest.raises(ValueError, match="still open"):
        guarded.launch(**setup.kwargs)
    assert not setup.calls


@pytest.mark.parametrize(
    "change",
    [
        "unknown_id",
        "duplicate_id",
        "pending",
        "provider_extra",
        "provider_tag",
        "not_active",
        "no_verifier",
        "mutable_checkout",
    ],
)
def test_parallel_admission_refuses_unprotected_or_ambiguous_work(guarded, setup, parallel, change):
    if change == "unknown_id":
        setup.kwargs["protected_active_instance_ids"] = ("9" * 32,)
    elif change == "duplicate_id":
        setup.kwargs["protected_active_instance_ids"] *= 2
    elif change == "pending":
        state = guarded.cloud.load_state(setup.state_path)
        state["pending_launch"] = {"name": "uncertain"}
        guarded.cloud.save_state(setup.state_path, state)
    elif change == "provider_extra":
        parallel.instances.append({**parallel.instances[0], "id": "9" * 32, "name": "scalar-pi-222222222222"})
    elif change == "provider_tag":
        parallel.instances[0]["tags"][0]["value"] = "different"
    elif change == "not_active":
        parallel.instances[0]["status"] = "terminating"
    elif change == "no_verifier":
        setup.kwargs["verify_active_leases"] = None
    else:
        setup.kwargs["existing_checkout"] = False
    with pytest.raises(ValueError, match="Protected|recorded|existing project|tags differ|verifier|checkout"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_sha256", "0" * 64),
        ("function_version_arn", "changed"),
        ("lease_id", "changed"),
        ("deadline_unix", NOW - 1),
        ("key_expires_unix", NOW),
        ("instance_id", "9" * 32),
    ],
)
def test_cleanup_proof_mismatch_prevents_post(guarded, setup, parallel, field, value):
    parallel.receipt["runs"][0][field] = value
    with pytest.raises(ValueError, match="Cleanup|cleanup"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


@pytest.mark.parametrize("observed", [NOW - 31, NOW + 1, float("nan")])
def test_stale_or_invalid_cleanup_verification_prevents_post(guarded, setup, parallel, observed):
    parallel.receipt["observed_unix"] = observed
    with pytest.raises(ValueError, match="verification"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_second_cleanup_verification_failure_prevents_post(guarded, setup, parallel):
    verifier = setup.kwargs["verify_active_leases"]

    def verify(opened):
        if parallel.verifications:
            raise ValueError("AWS schedule became disabled")
        return verifier(opened)

    setup.kwargs["verify_active_leases"] = verify
    with pytest.raises(ValueError, match="became disabled"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)
    assert "pending_launch" not in guarded.cloud.load_state(setup.state_path)


def test_suspend_after_pending_invalidates_parallel_verification(guarded, setup, parallel, monkeypatch):
    clock = [NOW]
    setup.kwargs["now"] = lambda: clock[0]
    save = guarded.cloud.save_state

    def suspend(path, state):
        save(path, state)
        if state.get("pending_launch"):
            clock[0] += 31

    monkeypatch.setattr(guarded.cloud, "save_state", suspend)
    with pytest.raises(ValueError, match="became stale"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)
    assert "pending_launch" not in guarded.cloud.load_state(setup.state_path)


def test_parallel_reservation_cannot_be_ignored_by_phase_budget(guarded, setup, parallel):
    setup.config["qualification_budget_usd"] = 100
    with pytest.raises(ValueError, match="phase allowance"):
        guarded.launch(**setup.kwargs)
    assert not any(call[1] == "POST" for call in setup.calls)


def test_readonly_research_budget_is_verified_without_republishing(guarded, setup):
    body = guarded.build_body(
        setup.config,
        guarded.cloud.load_state(setup.state_path),
        setup.armed,
        "true\n",
        b'{"budget":"exact bytes"}',
        rate=3.79,
        forecast_rate=3.79,
        existing_checkout=True,
        normalization_sha256="c" * 64,
    )
    bootstrap = yaml.safe_load(body["user_data"])["runcmd"][-1][2]
    worker = base64.b64decode(re.findall(r"printf '%s' '([A-Za-z0-9+/=]+)'", bootstrap)[1]).decode()
    assert "SCALAR_PI_VERIFY_BUDGET" in worker
    assert "base64 --decode > /lambda/nfs/scalar-pi/data/experiments/reports/budget.json" not in worker
    assert hashlib.sha256(b'{"budget":"exact bytes"}').hexdigest() in worker
