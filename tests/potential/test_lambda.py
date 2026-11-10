"""Cloud-control tests use fake APIs; none can provision a real resource."""

import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shlex
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cloud():
    spec = importlib.util.spec_from_file_location("scalar_pi_lambda_test", ROOT / "scripts/lambda/cloud.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config(tmp_path):
    return {
        "region": "us-west-3",
        "instance_type": "gpu_1x_h100_pcie",
        "image_id": "a" * 32,
        "operator_cidr": "203.0.113.1/32",
        "ssh_key_path": str(tmp_path / "ssh_key"),
        "max_hourly_rate_usd": 3.29,
        "budget_usd": 2000,
        "qualification_budget_usd": 100,
        "noncompute_spent_usd": 0,
        "noncompute_reserve_usd": 100,
        "credits_verified": True,
    }


class FakeLambda:
    def __init__(self, config):
        self.calls = []
        self.config = config
        self.status = "active"
        self.name = "scalar-pi-owned"

    def request(self, method, endpoint, body=None):
        self.calls.append((method, endpoint, body))
        if endpoint == "instance-types":
            return {
                self.config["instance_type"]: {
                    "instance_type": {"price_cents_per_hour": 329, "specs": {"gpus": 1}, "architecture": "x86_64"},
                    "regions_with_capacity_available": [{"name": "us-west-3"}],
                }
            }
        if endpoint == "images":
            return [
                {"id": "a" * 32, "architecture": "x86_64", "family": "gpu-base-22-04", "region": {"name": "us-west-3"}}
            ]
        if endpoint == "instance-operations/launch":
            self.name = body["name"]
            return {"instance_ids": ["owned-id"]}
        if endpoint == "instance-operations/terminate":
            self.status = "terminated"
            return {}
        if endpoint == "instances/owned-id":
            return {"name": self.name, "status": self.status, "ip": "203.0.113.10"}
        raise AssertionError((method, endpoint))


def test_dry_run_does_not_load_credentials_or_create_state(cloud, config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "lambda.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(cloud, "LambdaClient", lambda *_: pytest.fail("Dry run authenticated"))
    assert cloud.main(["--config", str(path), "prepare"]) == 0
    assert json.loads(capsys.readouterr().out)["resources_created"] == 0
    assert list(tmp_path.iterdir()) == [path]


def test_termination_never_adopts_unrecorded_id(cloud, config, tmp_path):
    client = FakeLambda(config)
    with pytest.raises(ValueError, match="not uniquely recorded"):
        cloud.terminate(client, {"runs": []}, tmp_path / "state.json", "someone-elses-id")
    assert client.calls == []


def test_termination_requires_both_id_and_name(cloud, config, tmp_path):
    client = FakeLambda(config)
    state = {"runs": [{"instance_id": "owned-id", "name": "different"}]}
    with pytest.raises(ValueError, match="identity"):
        cloud.terminate(client, state, tmp_path / "state.json", "owned-id")
    assert all(call[0] == "GET" for call in client.calls)


def test_confirmed_termination_is_idempotent_and_preserves_cost(cloud, config, tmp_path):
    client = FakeLambda(config)
    run = {
        "instance_id": "owned-id",
        "name": "scalar-pi-owned",
        "started_unix": 100,
        "ended_unix": 160,
        "hourly_rate_usd": 3.29,
        "estimated_compute_usd": 3.29 / 60,
    }
    state = {"runs": [run.copy()]}
    cloud.terminate(client, state, tmp_path / "state.json", "owned-id")
    assert state["runs"] == [run]
    assert client.calls == []


@pytest.mark.parametrize("error", [RuntimeError("transport failed"), KeyboardInterrupt()])
def test_uncertain_launch_is_not_retried_and_its_owned_instance_is_terminated(cloud, config, tmp_path, error):
    class InterruptedLambda(FakeLambda):
        def request(self, method, endpoint, body=None):
            if endpoint == "instances":
                self.calls.append((method, endpoint, body))
                return [{"id": "owned-id", "name": self.name}]
            result = super().request(method, endpoint, body)
            if endpoint == "instance-operations/launch":
                raise error
            return result

    client = InterruptedLambda(config)
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(max_hours=1, stage="qualification", budget_report=None, commit="a" * 40, script=script)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"
    assert "pending_launch" not in state
    assert state["runs"][0]["ended_unix"] >= state["runs"][0]["started_unix"]
    assert sum(endpoint == "instance-operations/launch" for _, endpoint, _ in client.calls) == 1


@pytest.mark.skipif(os.name != "posix", reason="The remote receiver runs in Linux bash")
def test_completed_session_receiver_preserves_script_and_cleanup(cloud, config, tmp_path, monkeypatch):
    class CapturedInput(io.BytesIO):
        def close(self):
            self.payload = self.getvalue()
            super().close()

    calls = []
    stdin = CapturedInput()

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdin=stdin, poll=lambda: 0, returncode=0)

    client = FakeLambda(config)
    script = tmp_path / "phase.sh"
    script.write_text("cat >/dev/null\nprintf 'receiver-ok\\n'\n")
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(max_hours=1, stage="qualification", budget_report=None, commit="a" * 40, script=script)
    real_run = cloud.subprocess.run
    monkeypatch.setenv("LAMBDA_API_KEY", "fixture-secret")
    monkeypatch.setattr(cloud.subprocess, "Popen", popen)
    monkeypatch.setattr(
        cloud.subprocess, "run", lambda argv, **kwargs: (calls.append((argv, kwargs)) or SimpleNamespace(returncode=0))
    )
    cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"
    assert state["runs"][0]["reports_downloaded"] is True
    assert "LAMBDA_API_KEY" not in calls[0][1]["env"]
    assert b"fixture-secret" not in stdin.payload
    ssh = next(argv for argv, _ in calls if argv[0] == "ssh" and argv[-1].startswith("bash -c "))
    transfer = next(argv for argv, _ in calls if argv[0] == "scp")
    assert transfer[:2] == ["scp", "-r"]
    assert transfer[2:-2] == ssh[1:-2]
    # A process reading stdin must not swallow the remainder of the transmitted
    # shell program. Run only the receiver locally, with a harmless probe script.
    monkeypatch.undo()
    result = real_run(shlex.split(ssh[-1]), input=script.read_text(), text=True, capture_output=True, check=True)
    assert result.stdout == "receiver-ok\n"


def test_price_increase_below_configured_ceiling_still_stops_budgeted_session(cloud, config, tmp_path, monkeypatch):
    class RisingPriceLambda(FakeLambda):
        quotes = 0

        def request(self, method, endpoint, body=None):
            result = super().request(method, endpoint, body)
            if endpoint == "instance-types":
                self.quotes += 1
                if self.quotes > 1:
                    result[config["instance_type"]]["instance_type"]["price_cents_per_hour"] = 400
            return result

    class RunningProcess:
        stdin = io.BytesIO()
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout):
            return self.returncode

    elapsed = [1000]
    monkeypatch.setattr(cloud.time, "time", lambda: elapsed[0])
    monkeypatch.setattr(cloud.time, "sleep", lambda _: elapsed.__setitem__(0, elapsed[0] + 61))
    process = RunningProcess()
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *_a, **_k: process)
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    config["max_hourly_rate_usd"] = 5
    client = RisingPriceLambda(config)
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(max_hours=1, stage="qualification", budget_report=None, commit="a" * 40, script=script)
    with pytest.raises(RuntimeError, match="budgeted rate"):
        cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert process.returncode == -15
    assert client.status == "terminated"
    assert state["runs"][0]["hourly_rate_usd"] == 4
    assert state["runs"][0]["estimated_compute_usd"] == pytest.approx(4 * 2 / 60)


def test_failed_ssh_still_terminates_and_records_cost(cloud, config, tmp_path, monkeypatch):
    client = FakeLambda(config)
    script = tmp_path / "phase.sh"
    script.write_text("exit 3\n")
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(max_hours=1, stage="qualification", budget_report=None, commit="a" * 40, script=script)
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fixture")))
    with pytest.raises(OSError, match="fixture"):
        cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"
    assert state["runs"][0]["ended_unix"] >= state["runs"][0]["started_unix"]
    assert any(method == "POST" and endpoint.endswith("terminate") for method, endpoint, _ in client.calls)


def test_ssh_readiness_timeout_still_terminates_instance(cloud, config, tmp_path, monkeypatch):
    client = FakeLambda(config)
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(max_hours=1, stage="qualification", budget_report=None, commit="a" * 40, script=script)
    elapsed = [1000]
    monkeypatch.setattr(cloud.time, "time", lambda: elapsed[0])
    monkeypatch.setattr(cloud.time, "sleep", lambda _: elapsed.__setitem__(0, elapsed[0] + 61))
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=255))
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Started work before SSH was ready"))
    with pytest.raises(TimeoutError, match="SSH did not become ready"):
        cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"
    assert state["runs"][0]["ended_unix"] == elapsed[0]


def test_launch_never_uploads_api_key(cloud, config, monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "fixture-secret-never-upload")
    payload = cloud.launch_body(config, {}, "a" * 40, "scalar-pi-owned")
    assert "fixture-secret" not in json.dumps(payload)
    assert "203.0.113.1/32" in payload["user_data"]
    assert "8000" not in payload["user_data"]
    assert "0.0.0.0/0" not in payload["user_data"]


def test_phase_allowance_includes_previous_sessions_and_headroom(cloud, config):
    state = {"runs": [{"started_unix": 0, "ended_unix": 3600, "hourly_rate_usd": 3.29, "stage": "qualification"}]}
    report = cloud.check_allowance(config, state, 1, 3.29, "qualification")
    assert report["spent_usd"] == 3.29
    with pytest.raises(ValueError, match="phase allowance"):
        cloud.check_allowance(config, state, 25, 3.29, "qualification")
    with pytest.raises(ValueError, match="measured"):
        cloud.check_allowance(config, state, 1, 3.29, "research")
    with pytest.raises(ValueError, match="finite"):
        cloud.check_allowance(config, state, float("nan"), 3.29, "qualification")


def test_os_lock_rejects_concurrent_controller(cloud, tmp_path):
    lock = tmp_path / "controller.lock"
    with cloud.project_lock(lock), pytest.raises(RuntimeError, match="Another project"), cloud.project_lock(lock):
        pytest.fail("Second controller obtained lock")
    with cloud.project_lock(lock):
        pass


def test_matrix_paths_work_on_host_and_in_containers(tmp_path):
    renderer = runpy.run_path(str(ROOT / "scripts/lambda/matrix.py"))
    full = renderer["render_matrix"](ROOT, tmp_path, "full")
    pilot = renderer["render_matrix"](ROOT, tmp_path, "pilot")
    assert full["training_seeds"] == [17, 42, 93]
    assert pilot["training_seeds"] == [42]
    assert len(full["variants"]) == 10
    assert len(pilot["variants"]) == 4
    assert all(str(tmp_path) in variant["checkpoint_dir"] for variant in pilot["variants"])
    assert "server" in pilot["server_command_prefix"]


def test_budget_selects_full_then_complete_pilot_then_stops():
    preflight = runpy.run_path(str(ROOT / "scripts/potential_preflight.py"))
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    profile = {
        "measured_gpu": True,
        "model_kind": "pi05_full",
        "hardware": "TEST ONLY",
        "measurement_source": "test fixture",
        "evaluation_measurement_source": "test fixture",
        "warmup_optimizer_updates": 10,
        "timed_optimizer_updates": 100,
        "effective_batch_size": 256,
        "optimizer_updates": 30000,
        "seconds_per_optimizer_update": {"B0": 1, "B1": 1, "P": 1},
        "checkpoint_seconds_per_save": {"B0": 1, "B1": 1, "P": 1},
        "checkpoint_interval_updates": {"B0": 5000, "B1": 5000, "P": 5000},
        "fixed_cost_usd": 100,
        "evaluation_seconds_per_episode": {
            group: {steps: dict.fromkeys(suites, 1) for steps in ("1", "10")}
            for group in ("B0", "B1", "P", "reference")
        },
    }
    choose = preflight["choose_study"]
    assert choose(profile, 3.29, 2000)["selected_study"] == "full"
    pilot = choose(profile, 3.29, 300)
    assert pilot["selected_study"] == "pilot"
    assert pilot["training_seeds"] == [42]
    assert pilot["optimizer_updates"] == 30000
    assert pilot["studies"]["pilot"]["evaluation_gpu_hours"] == pytest.approx(16000 / 3600)
    assert choose(profile, 3.29, 200)["selected_study"] is None
    del profile["evaluation_seconds_per_episode"]["P"]["10"]["libero_10"]
    with pytest.raises(ValueError, match="every native"):
        choose(profile, 3.29, 2000)
