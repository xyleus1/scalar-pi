"""Synthetic-only private launch wrapper tests; every remote service is mocked."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "scripts/lambda"
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("replica_preflight_test", HERE / "replicated_preflight_launch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
NOW = 1_800_000_000
OLD = "e" * 32
NEW = "b" * 32
WORKSPACE = "c" * 32


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    config = {
        "region": "us-southeast-1",
        "storage_mode": "replicated_local",
        "instance_type": "gpu_2x_h100_sxm5",
        "image_id": "a" * 32,
        "operator_cidr": "203.0.113.1/32",
        "max_hourly_rate_usd": 9.58,
        "budget_usd": 5000,
        "qualification_budget_usd": 500,
        "noncompute_spent_usd": 0,
        "noncompute_reserve_usd": 100,
        "credits_verified": True,
        "full_study_authorized": True,
        "full_study_authorization_evidence": "fixture explicit full study",
    }
    authorization = {
        "full_study_authorized": True,
        "backup_gpus_authorized": True,
        "seeds": [42, 17, 93],
        "arms": ["B0", "B1", "P"],
        "optimizer_updates_per_arm_seed": 30000,
        "evaluation_episodes": 40000,
    }
    plan = {
        "schema_version": 1,
        "attempt": "scalar-pi-0123456789ab",
        "controller_commit": "d" * 40,
        "worker_commit": m.WORKER_COMMIT,
        "workspace_id": WORKSPACE,
        "initial_worker_purpose": "host_preflight_only",
        "admission_policy": "replicated_host_preflight_only",
        "serialized_github_admission": True,
        "deadline_utc": m.dt.datetime.fromtimestamp(NOW + 3600, m.dt.UTC).isoformat(),
    }
    for name, value in (
        ("config", config),
        ("authorization", authorization),
        (
            "github_cleanup_receipt",
            {
                "status": "installed_in_github_environment",
                "expires_at": m.dt.datetime.fromtimestamp(NOW + 10000, m.dt.UTC).isoformat(),
            },
        ),
    ):
        plan[name] = write(tmp_path / (name + ".json"), value)
    worker = tmp_path / m.WORKER_NAME
    worker.write_bytes((HERE / m.WORKER_NAME).read_bytes())
    plan["worker_script"] = {
        "path": str(worker),
        "sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
    }
    plan["protected_workers"] = [
        {
            "instance_id": OLD,
            "launch_receipt": write(tmp_path / "old-launch.json", {"instance_id": OLD}),
            "lease_metadata": write(tmp_path / "old-lease.json", {"instance_id": OLD}),
        }
    ]
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    state = {
        "schema_version": 1,
        "resources": {},
        "runs": [{"instance_id": OLD, "name": "scalar-pi-111111111111"}],
    }
    m.cloud.save_state(tmp_path / "lambda-state.json", state)
    pointer = tmp_path / "latest-remote-launch.json"
    pointer.write_bytes(b"existing current pointer must remain unchanged")
    write(
        tmp_path / "cloud-service-identities-20261116.json",
        {
            "identities": {
                "launcher": {
                    "identity": {"id": "launcher"},
                    "membership": {"workspace_id": WORKSPACE},
                }
            }
        },
    )
    calls = []
    role = {
        "id": m.guarded.LAUNCH_ROLE_ID,
        "name": "workspace-member",
        "permission_ids": ["compute:instance:launch"],
    }

    class Reader:
        key = "fixture-master-key"

        def request(self, method, endpoint, body=None):
            calls.append((method, endpoint, body))
            if endpoint == "instances":
                return []
            if endpoint.endswith("/role-assignments"):
                return [
                    {
                        "role_id": role["id"],
                        "scope": {"type": "workspace", "workspace_id": WORKSPACE},
                    }
                ]
            if endpoint.endswith("/groups"):
                return []
            if endpoint.endswith("/workspaces"):
                return [{"id": WORKSPACE}]
            if endpoint == "roles/" + role["id"]:
                return role
            if method == "POST":
                assert endpoint == "identities/launcher/api-keys"
                pending = m.read(tmp_path / "extra-workers" / plan["attempt"] / "launcher-key-receipt.json")
                assert pending["status"] == "creation_pending"
                return {
                    "id": "key-id",
                    "expires_at": m.dt.datetime.fromtimestamp(NOW + 300, m.dt.UTC).isoformat(),
                    "api_key": "fixture-creator-key",
                }
            assert method == "DELETE"
            assert endpoint == "api-keys/key-id"
            return {}

    class Guard:
        def dispatch(self, request, *, dry_run=False):
            calls.append(("dispatch", request, dry_run))
            path = tmp_path / "github-guards" / plan["attempt"] / "guard.json"
            write(path, {"run_id": "5678", "request": request})
            return path

        def wait_ready(self, path, timeout_seconds):
            return {"launch_not_after_unix": NOW + 900, "github": {"run_id": "5678"}}

    forwarded = []

    def guarded_launch(*args, **kwargs):
        forwarded.append((args, kwargs))
        assert args[1] == tmp_path / "lambda-state.json"
        assert kwargs["stage"] == "qualification"
        assert kwargs["budget_report_path"] is None
        assert kwargs["protected_active_instance_ids"] == (OLD,)
        assert kwargs["existing_checkout"] is False
        assert "normalization_sha256" not in kwargs
        assert args[5]["storage_mode"] == "replicated_local"
        return {
            "instance_id": NEW,
            "watchdog": {"github": {"run_id": "5678"}},
            "deadline_unix": NOW + 3600,
        }

    monkeypatch.setattr(m.cloud, "select_type", lambda *a: 9.58)
    monkeypatch.setattr(m.cloud, "check_allowance", lambda *a, **k: {"checked": True})
    monkeypatch.setattr(m.guarded, "verify_instances", lambda *a, **k: None)
    monkeypatch.setattr(m.guarded, "protected_coverage", lambda *a, **k: ({"checked": True}, {}))
    monkeypatch.setattr(m.guarded, "launch", guarded_launch)
    helpers = SimpleNamespace(
        audit_identity=lambda *a: ([{"role_id": role["id"]}], [role]),
        verify_github_key=lambda *a: calls.append(("verify GitHub cleanup key", None, None)),
    )
    return SimpleNamespace(
        root=tmp_path,
        plan=plan,
        plan_path=plan_path,
        reader=Reader(),
        guard=Guard(),
        calls=calls,
        forwarded=forwarded,
        pointer=pointer,
        helpers=helpers,
    )


def execute(case):
    return m.launch(
        case.plan_path,
        root=case.root,
        reader=case.reader,
        guard=case.guard,
        verifier=lambda _: {},
        helpers=case.helpers,
        now=lambda: NOW,
    )


def test_plan_is_local_and_needs_no_measured_report(case):
    m.load_plan(case.plan_path, root=case.root, now=lambda: NOW)
    assert case.calls == []


@pytest.mark.parametrize("key", ["study_plan", "budget_report", "selected_study"])
def test_measured_study_field_is_rejected(case, key):
    case.plan[key] = None
    write(case.plan_path, case.plan)
    with pytest.raises(ValueError, match="measured study"):
        execute(case)
    assert case.calls == []


def test_changed_worker_is_rejected_before_cloud_calls(case):
    Path(case.plan["worker_script"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="Pinned input changed"):
        execute(case)
    assert case.calls == []


def test_launch_only_writes_own_attempt_and_revokes_exact_creator(case):
    result = execute(case)
    assert result["instance_id"] == NEW
    assert "assignment" not in result
    assert not (case.root / "extra-workers/assignments").exists()
    assert case.pointer.read_bytes() == b"existing current pointer must remain unchanged"
    assert case.calls[-1] == ("DELETE", "api-keys/key-id", None)
    folder = case.root / "extra-workers" / case.plan["attempt"]
    assert m.read(folder / "launch.json")["status"] == "launched"
    assert case.forwarded[0][0][2] == folder / "worker_script.sh"
    assert case.forwarded[0][1]["budget_report_path"] is None
    for path in folder.rglob("*"):
        if path.is_file():
            assert "fixture-creator-key" not in path.read_text()
            assert "fixture-master-key" not in path.read_text()


def test_launch_is_never_redispatched_automatically(case):
    execute(case)
    before = copy.deepcopy(case.calls)
    with pytest.raises(ValueError, match="already recorded"):
        execute(case)
    assert case.calls == before


def test_uncertain_launch_retains_intent_and_revokes_key(case, monkeypatch):
    def uncertain(*a, **k):
        raise RuntimeError("launch uncertain")

    monkeypatch.setattr(m.guarded, "launch", uncertain)
    with pytest.raises(RuntimeError, match="uncertain"):
        execute(case)
    assert case.calls[-1] == ("DELETE", "api-keys/key-id", None)
    folder = case.root / "extra-workers" / case.plan["attempt"]
    assert m.read(folder / "launch.json")["status"] == "arming"
    with pytest.raises(ValueError, match="already recorded"):
        execute(case)


def test_expired_github_cleanup_key_blocks_before_dispatch(case):
    path = Path(case.plan["github_cleanup_receipt"]["path"])
    data = m.read(path)
    data["expires_at"] = m.dt.datetime.fromtimestamp(NOW + 4000, m.dt.UTC).isoformat()
    case.plan["github_cleanup_receipt"] = write(path, data)
    write(case.plan_path, case.plan)
    with pytest.raises(ValueError, match="cleanup key expires"):
        execute(case)
    assert case.calls == []


@pytest.mark.parametrize("delta", [0, -1, 10801, 14400])
def test_qualification_duration_bound(case, delta):
    case.plan["deadline_utc"] = m.dt.datetime.fromtimestamp(NOW + delta, m.dt.UTC).isoformat()
    write(case.plan_path, case.plan)
    with pytest.raises(ValueError):
        execute(case)
    assert case.calls == []


def test_exact_three_hour_qualification_is_allowed(case):
    case.plan["deadline_utc"] = m.dt.datetime.fromtimestamp(NOW + 10800, m.dt.UTC).isoformat()
    cleanup = m.read(case.plan["github_cleanup_receipt"]["path"])
    cleanup["expires_at"] = m.dt.datetime.fromtimestamp(NOW + 20000, m.dt.UTC).isoformat()
    case.plan["github_cleanup_receipt"] = write(Path(case.plan["github_cleanup_receipt"]["path"]), cleanup)
    write(case.plan_path, case.plan)
    m.load_plan(case.plan_path, root=case.root, now=lambda: NOW)
    assert not case.calls


def test_wrong_approved_worker_hash_cannot_pass_even_if_self_consistent(case):
    path = Path(case.plan["worker_script"]["path"])
    path.write_text("printf changed", encoding="utf-8")
    case.plan["worker_script"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(case.plan_path, case.plan)
    with pytest.raises(ValueError, match="reviewed local-storage"):
        execute(case)
    assert not case.calls


def test_failed_protected_guard_blocks_before_dispatch(case, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("Protected cleanup guard not live")

    monkeypatch.setattr(m.guarded, "protected_coverage", failed)
    with pytest.raises(ValueError, match="guard not live"):
        execute(case)
    assert not any(call[0] in {"POST", "dispatch"} for call in case.calls)


def test_failed_ready_guard_creates_no_creator_key(case, monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("GitHub guard not live")

    monkeypatch.setattr(case.guard, "wait_ready", failed)
    with pytest.raises(ValueError, match="guard not live"):
        execute(case)
    assert not any(call[0] == "POST" for call in case.calls)


def test_key_receipt_write_failure_still_revokes_key(case, monkeypatch):
    save = m.cloud.save_state

    def failed(path, value):
        if Path(path).name == "launcher-key-receipt.json" and "key_sha256" in value:
            raise OSError("fixture disk failure")
        return save(path, value)

    monkeypatch.setattr(m.cloud, "save_state", failed)
    with pytest.raises(OSError, match="disk failure"):
        execute(case)
    assert case.calls[-1] == ("DELETE", "api-keys/key-id", None)
    assert not case.forwarded


def test_receipt_explicitly_records_preflight_not_training(case):
    result = execute(case)
    assert result["admission_policy"] == "replicated_host_preflight_only"
    assert result["stage"] == "qualification"
    assert result["measured_study_report"] is None
    assert result["initial_worker_purpose"] == "host_preflight_only"
    args, kwargs = case.forwarded[0]
    assert m.read(args[3])["workspace_ids"] == [WORKSPACE]
    assert kwargs["budget_report_path"] is None
    assert not list((case.root / "extra-workers").rglob("study_plan.json"))


def test_actual_cli_preview_needs_no_cloud_credentials(case):
    now = int(time.time())
    case.plan["deadline_utc"] = m.dt.datetime.fromtimestamp(now + 3600, m.dt.UTC).isoformat()
    cleanup = m.read(case.plan["github_cleanup_receipt"]["path"])
    cleanup["expires_at"] = m.dt.datetime.fromtimestamp(now + 12000, m.dt.UTC).isoformat()
    case.plan["github_cleanup_receipt"] = write(Path(case.plan["github_cleanup_receipt"]["path"]), cleanup)
    write(case.plan_path, case.plan)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(secret in key.upper() for secret in ("TOKEN", "SECRET", "ACCESS_KEY", "API_KEY"))
    }
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "replicated_preflight_launch.py"),
            "--plan",
            str(case.plan_path),
            "--control-root",
            str(case.root),
            "--aws-plan",
            str(case.root / "not-loaded-in-preview.json"),
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "prepared_no_calls"
    assert not (case.root / "extra-workers").exists()


def test_private_import_contract_in_fresh_process_preserves_portable_modules():
    control = Path.home() / ".config/scalar-pi"
    if not all((control / name).is_file() for name in m.PRIVATE_PINS):
        pytest.skip("Operator-only private helpers are not installed on CI")
    source = (
        "import sys; from pathlib import Path; "
        f"sys.path.insert(0,{str(HERE)!r}); import replicated_preflight_launch as m; "
        "before=list(sys.path); h=m.private_helpers(m.PRIVATE); "
        "assert sys.path==before; assert h.cloud is m.cloud and h.guarded is m.guarded; "
        "assert Path(m.guarded.__file__).resolve().parent==Path(m.__file__).resolve().parent; "
        "assert callable(h.audit_identity) and callable(h.verify_github_key); print('imports_bound_no_calls')"
    )
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imports_bound_no_calls"


def test_unknown_key_creation_result_is_not_repeated(case, monkeypatch):
    original = case.reader.request

    def failed(method, endpoint, body=None):
        if method == "POST":
            case.calls.append((method, endpoint, body))
            raise TimeoutError("key creation response lost")
        return original(method, endpoint, body)

    monkeypatch.setattr(case.reader, "request", failed)
    with pytest.raises(TimeoutError, match="key creation response lost"):
        execute(case)
    receipt = case.root / "extra-workers" / case.plan["attempt"] / "launcher-key-receipt.json"
    assert m.read(receipt)["status"] == "creation_pending"
    with pytest.raises(ValueError, match="already recorded"):
        execute(case)
    assert sum(call[0] == "POST" for call in case.calls) == 1
    assert not any(call[0] == "DELETE" for call in case.calls)


def test_embedded_host_probe_compiles_and_never_starts_science():
    script = (HERE / m.WORKER_NAME).read_text()
    source = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(source, m.WORKER_NAME, "exec")
    assert "minimum = expected_count * 200 * 1024**3" in source
    assert "host_observed_not_qualified" in source
    assert "train_potential.py" not in source and "evaluate_potential.py" not in source


@pytest.mark.parametrize(("expected", "actual"), [(2, 2), (4, 4), (4, 2), (2, 4), (3, 3)])
def test_host_probe_checks_actual_physical_gpu_count(expected, actual):
    script = (HERE / m.WORKER_NAME).read_text()
    source = script.split("python3 - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "validate_gpus"
    )
    namespace = {"re": re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), m.WORKER_NAME, "exec"), namespace)
    rows = [f"NVIDIA H100 80GB HBM3, 81559, 580.196.07, GPU-{index:036x}, 9.0" for index in range(actual)]
    validate = namespace["validate_gpus"]
    if expected != actual or expected not in (2, 4):
        with pytest.raises(AssertionError):
            validate(rows, expected)
    else:
        assert len(validate(rows, expected)) == actual
        with pytest.raises(AssertionError, match="Duplicate"):
            validate([rows[0]] * actual, expected)
