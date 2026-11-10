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
import yaml

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
        self.price_cents_per_hour = 329

    def request(self, method, endpoint, body=None):
        self.calls.append((method, endpoint, body))
        if endpoint == "instance-types":
            return {
                self.config["instance_type"]: {
                    "instance_type": {
                        "price_cents_per_hour": self.price_cents_per_hour,
                        "specs": {"gpus": 1},
                        "architecture": "x86_64",
                    },
                    "regions_with_capacity_available": [{"name": self.config["region"]}],
                }
            }
        if endpoint == "images":
            return [
                {
                    "id": "a" * 32,
                    "architecture": "x86_64",
                    "family": "gpu-base-22-04",
                    "region": {"name": self.config["region"]},
                }
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


def test_api_requests_identify_the_controller(cloud, monkeypatch):
    requests = []

    def open_request(request, **_kwargs):
        requests.append(request)
        return io.BytesIO(b'{"data": []}')

    monkeypatch.setattr(cloud.urllib.request, "build_opener", lambda *_args: SimpleNamespace(open=open_request))
    assert cloud.LambdaClient("fixture-secret").request("GET", "instances") == []
    assert requests[0].get_header("User-agent") == "scalar-pi/1.0 (+https://github.com/xyleus1/scalar-pi)"
    assert requests[0].get_header("Accept") == "application/json"


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
@pytest.mark.parametrize("returncode", [0, 3])
def test_completed_session_receiver_preserves_script_and_cleanup(cloud, config, tmp_path, monkeypatch, returncode):
    class CapturedInput(io.BytesIO):
        def close(self):
            self.payload = self.getvalue()
            super().close()

    calls = []
    stdin = CapturedInput()

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdin=stdin, poll=lambda: returncode, returncode=returncode)

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
    if returncode:
        with pytest.raises(RuntimeError, match="Remote script failed"):
            cloud.run_session(client, config, state, tmp_path / "state.json", args)
    else:
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
    assert state["runs"][0]["reports_downloaded"] is True


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


def test_report_ledger_write_failure_still_terminates_instance(cloud, config, tmp_path, monkeypatch):
    client = FakeLambda(config)
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    args = SimpleNamespace(max_hours=1, stage="qualification", budget_report=None, commit="a" * 40, script=script)
    save_state = cloud.save_state

    def fail_report_save(path, ledger):
        if ledger["runs"] and ledger["runs"][0].get("reports_downloaded") and not ledger["runs"][0].get("ended_unix"):
            raise OSError("Cannot persist report status")
        save_state(path, ledger)

    monkeypatch.setattr(cloud, "save_state", fail_report_save)
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(
        cloud.subprocess,
        "Popen",
        lambda *_a, **_k: SimpleNamespace(stdin=io.BytesIO(), poll=lambda: 0, returncode=0),
    )
    with pytest.raises(OSError, match="Cannot persist report status"):
        cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"
    persisted = json.loads((tmp_path / "state.json").read_text())["runs"][0]
    assert persisted["ended_unix"] >= persisted["started_unix"]
    assert persisted["reports_downloaded"] is True


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


@pytest.fixture
def canary_session():
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(
        max_hours=0.25,
        stage="canary",
        budget_report=None,
        commit="a" * 40,
        script=ROOT / "scripts/lambda/canary.sh",
    )
    return state, args


@pytest.mark.parametrize(("ready_after", "timeout_first"), [(0, False), (180, False), (870, True)])
def test_canary_uses_ssh_readiness_while_provider_still_reports_booting(
    cloud, canary_config, canary_session, tmp_path, monkeypatch, ready_after, timeout_first
):
    state, args = canary_session
    client = FakeLambda(canary_config)
    client.status = "booting"
    elapsed = [1000]
    readiness_calls = []
    submitted = []

    class CapturedInput(io.BytesIO):
        def close(self):
            submitted.append(self.getvalue().decode())
            super().close()

    def remote_command(argv, **kwargs):
        if argv[0] == "scp":
            return SimpleNamespace(returncode=0)
        assert argv[-1] == "true"
        readiness_calls.append(elapsed[0])
        if timeout_first and len(readiness_calls) == 1:
            elapsed[0] += kwargs["timeout"]
            raise cloud.subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=0 if elapsed[0] >= 1000 + ready_after else 255)

    def start_worker(*_args, **_kwargs):
        assert client.status == "booting"
        return SimpleNamespace(stdin=CapturedInput(), poll=lambda: 0, returncode=0)

    monkeypatch.setattr(cloud.time, "time", lambda: elapsed[0])
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(cloud.subprocess, "run", remote_command)
    monkeypatch.setattr(cloud.subprocess, "Popen", start_worker)
    cloud.run_session(client, canary_config, state, tmp_path / "state.json", args)
    assert 1000 + ready_after <= elapsed[0] < 1900
    assert len(submitted) == 1
    assert submitted[0].index("sudo cloud-init status --wait") < submitted[0].index("scripts/lambda/canary.py probe")
    assert "git submodule update" not in submitted[0]
    assert state["runs"][0]["reports_downloaded"] is True
    assert client.status == "terminated"


def test_canary_ssh_timeouts_use_original_session_deadline_and_terminate(
    cloud, canary_config, canary_session, tmp_path, monkeypatch
):
    state, args = canary_session
    client = FakeLambda(canary_config)
    client.status = "booting"
    elapsed = [1000]

    def timeout(argv, **kwargs):
        elapsed[0] += kwargs["timeout"]
        raise cloud.subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(cloud.time, "time", lambda: elapsed[0])
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(cloud.subprocess, "run", timeout)
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Started before SSH was ready"))
    with pytest.raises(TimeoutError, match="SSH did not become ready"):
        cloud.run_session(client, canary_config, state, tmp_path / "state.json", args)
    assert state["runs"][0]["ended_unix"] == 1900
    assert state["runs"][0]["reports_downloaded"] is False
    assert client.status == "terminated"


@pytest.mark.parametrize("status", ["unhealthy", "terminated", "preempted"])
def test_canary_rejects_terminal_provider_states_before_ssh(
    cloud, canary_config, canary_session, tmp_path, monkeypatch, status
):
    state, args = canary_session
    client = FakeLambda(canary_config)
    client.status = status
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: pytest.fail("Probed SSH on an unhealthy instance"))
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Started an unhealthy instance"))
    with pytest.raises(RuntimeError, match="did not become healthy"):
        cloud.run_session(client, canary_config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"


@pytest.mark.parametrize("stage", ["qualification", "research"])
def test_noncanary_stages_still_wait_for_active_status(cloud, config, tmp_path, monkeypatch, stage):
    client = FakeLambda(config)
    client.status = "booting"
    elapsed = [1000]
    script = tmp_path / "phase.sh"
    script.write_text("exit 0\n")
    report = tmp_path / "budget.json"
    report.write_text(
        json.dumps(
            {
                "selected_study": "pilot",
                "budget_usd": 2000,
                "live_hourly_rate_usd": 3.29,
                "forecast_hourly_rate_usd": 3.29,
                "remaining_compute_cost_with_headroom_usd": 1,
                "experiment_identity": {"code_commit": "a" * 40, "container_image_id": "sha256:" + "b" * 64},
                "study_lock_sha256": "c" * 64,
            }
        )
    )
    args = SimpleNamespace(
        max_hours=0.025,
        stage=stage,
        budget_report=report if stage == "research" else None,
        commit="a" * 40,
        script=script,
    )
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    monkeypatch.setattr(cloud.time, "time", lambda: elapsed[0])
    monkeypatch.setattr(cloud.time, "sleep", lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds))
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: pytest.fail("Probed noncanary SSH while booting"))
    monkeypatch.setattr(
        cloud.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Started noncanary work while booting")
    )
    with pytest.raises(TimeoutError, match="Session deadline reached during boot"):
        cloud.run_session(client, config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"


def test_launch_never_uploads_api_key(cloud, config, monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "fixture-secret-never-upload")
    payload = cloud.launch_body(config, {"resources": {"filesystem_id": "owned-fs"}}, "a" * 40, "scalar-pi-owned")
    assert "fixture-secret" not in json.dumps(payload)
    assert "203.0.113.1/32" in payload["user_data"]
    assert "8000" not in payload["user_data"]
    assert "0.0.0.0/0" not in payload["user_data"]
    assert payload["file_system_mounts"] == [{"file_system_id": "owned-fs", "mount_point": cloud.REMOTE_ROOT}]
    assert "file_system_names" not in payload


def test_cloud_init_configures_nvidia_runtime_after_firewall(cloud, config):
    payload = cloud.launch_body(config, {"resources": {"filesystem_id": "owned-fs"}}, "a" * 40, "scalar-pi-owned")
    bootstrap = yaml.safe_load(payload["user_data"])
    assert bootstrap["packages"] == ["ufw", "git"]
    assert bootstrap["runcmd"][:3] == [
        ["ufw", "default", "deny", "incoming"],
        ["ufw", "allow", "from", "203.0.113.1/32", "to", "any", "port", "22", "proto", "tcp"],
        ["ufw", "--force", "enable"],
    ]
    command = bootstrap["runcmd"][-1]
    assert command[:2] == ["bash", "-c"]
    script = command[2]
    assert script.startswith("set -euo pipefail\n")
    assert "if ! command -v nvidia-ctk >/dev/null || ! command -v nvidia-container-runtime >/dev/null; then" in script
    assert "https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list" in script
    assert "signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg" in script
    for package in (
        "nvidia-container-toolkit",
        "nvidia-container-toolkit-base",
        "libnvidia-container-tools",
        "libnvidia-container1",
    ):
        assert package + "=1.20.1-1" in script
    assert script.endswith("fi\nnvidia-ctk runtime configure --runtime=docker\nsystemctl restart docker\n")
    if os.name == "posix":
        cloud.subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True, check=True)


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


@pytest.mark.parametrize(("stage", "hours"), [("canary", 0.251), ("qualification", 2.01), ("research", 4.01)])
def test_supervised_sessions_have_stage_limits(cloud, config, stage, hours):
    with pytest.raises(ValueError, match="Supervised"):
        cloud.session_deadline(config, stage, hours, 1000)


def test_session_cannot_straddle_announced_price_change(cloud, config):
    config["price_change_at_utc"] = "2026-11-16T00:00:00+00:00"
    change = cloud.dt.datetime.fromisoformat(config["price_change_at_utc"]).timestamp()
    with pytest.raises(ValueError, match="price change"):
        cloud.session_deadline(config, "research", 4, change - 3600)
    assert cloud.session_deadline(config, "research", 4, change) == change + 4 * 3600


def test_forecast_price_and_storage_are_reserved_without_double_counting(cloud, config):
    config["forecast_hourly_rate_usd"] = 3.79
    report = {
        "selected_study": "pilot",
        "budget_usd": 2000,
        "live_hourly_rate_usd": 3.29,
        "forecast_hourly_rate_usd": 3.79,
        "remaining_compute_cost_with_headroom_usd": 1500,
        "remaining_noncompute_cost_usd": 200,
    }
    allowance = cloud.check_allowance(config, {"runs": []}, 4, 3.29, "research", report)
    assert allowance["session_ceiling_usd"] == pytest.approx(4 * 3.29)
    assert allowance["with_reserve_usd"] == pytest.approx(200 + 1.2 * 4 * 3.79)
    report["remaining_noncompute_cost_usd"] = 501
    with pytest.raises(ValueError, match="remaining balance"):
        cloud.check_allowance(config, {"runs": []}, 4, 3.29, "research", report)


@pytest.fixture
def canary_config(config):
    return {
        **config,
        "credits_verified": False,
        "forecast_hourly_rate_usd": 3.79,
        "canary_allowance_usd": 2,
        "canary_noncompute_reserve_usd": 0.75,
    }


def test_unverified_canary_has_cumulative_cost_and_reserve_limit(cloud, canary_config):
    allowance = cloud.check_allowance(canary_config, {"runs": []}, 0.25, 3.29, "canary")
    assert allowance["canary_with_reserve_usd"] == pytest.approx(1.887)
    assert canary_config["credits_verified"] is False
    state = {"runs": [{"started_unix": 0, "ended_unix": 900, "hourly_rate_usd": 3.29, "stage": "canary"}]}
    with pytest.raises(ValueError, match="Cumulative canary"):
        cloud.check_allowance(canary_config, state, 0.25, 3.29, "canary")
    canary_config["noncompute_spent_usd"] = 0.20
    with pytest.raises(ValueError, match="Cumulative canary"):
        cloud.check_allowance(canary_config, {"runs": []}, 0.25, 3.29, "canary")


def test_approved_retry_preserves_spending_and_reserve(cloud, canary_config):
    canary_config["canary_allowance_usd"] = 5
    state = {"runs": [{"started_unix": 0, "ended_unix": 17 * 60, "hourly_rate_usd": 3.29, "stage": "canary"}]}
    cloud.validate_config(canary_config)
    allowance = cloud.check_allowance(canary_config, state, 0.25, 3.29, "canary")
    assert allowance["canary_with_reserve_usd"] == pytest.approx(17 / 60 * 3.29 + 0.75 + 1.2 * 0.25 * 3.79)
    assert canary_config["credits_verified"] is False


def test_provider_charge_reconciles_budget_without_rewriting_elapsed_estimate(cloud, canary_config):
    canary_config.update(canary_allowance_usd=5, forecast_hourly_rate_usd=4.89, max_hourly_rate_usd=4.89)
    first = {"started_unix": 0, "ended_unix": 17 * 60, "hourly_rate_usd": 3.29, "stage": "canary"}
    first["estimated_compute_usd"] = cloud.session_cost(first)
    second = {"started_unix": 0, "ended_unix": 36 * 60, "hourly_rate_usd": 3.29, "stage": "canary"}
    state = {"runs": [first, second]}
    with pytest.raises(ValueError, match="Cumulative canary"):
        cloud.check_allowance(canary_config, state, 0.25, 4.29, "canary")
    first.update(provider_compute_upper_usd=0.13, provider_compute_source="fixture: retained provider Usage receipt")
    recorded = first.copy()
    allowance = cloud.check_allowance(canary_config, state, 0.25, 4.29, "canary")
    assert allowance["spent_usd"] == pytest.approx(0.13 + 1.974)
    assert allowance["canary_with_reserve_usd"] == pytest.approx(4.321)
    assert cloud.session_cost(first) == first["estimated_compute_usd"]
    assert first == recorded


@pytest.mark.parametrize(
    "invalid",
    [
        {"ended_unix": None},
        {"provider_compute_source": None},
        {"provider_compute_source": " "},
        {"provider_compute_upper_usd": -0.01},
        {"provider_compute_upper_usd": float("nan")},
        {"provider_compute_upper_usd": float("inf")},
        {"provider_compute_upper_usd": True},
        {"provider_compute_upper_usd": "0.13"},
    ],
)
def test_provider_charge_requires_closed_run_and_valid_evidence(cloud, invalid):
    run = {"ended_unix": 100, "provider_compute_upper_usd": 0.13, "provider_compute_source": "fixture"}
    with pytest.raises(ValueError, match="Provider compute accounting|provider_compute_upper_usd"):
        cloud.accounted_cost({**run, **invalid})


@pytest.mark.parametrize("stage", ["qualification", "research"])
def test_alternate_h100_profile_cannot_train_even_with_verified_billing(cloud, config, tmp_path, stage):
    config.update(instance_type="gpu_1x_h100_sxm5", region="us-south-2", credits_verified=True)
    cloud.validate_config(config)
    client = FakeLambda(config)
    with pytest.raises(ValueError, match="Alternate hardware is canary-only"):
        cloud.run_session(client, config, {"runs": []}, tmp_path / "state.json", SimpleNamespace(stage=stage))
    assert client.calls == []


def test_regional_prepare_keeps_retained_filesystem_and_creates_named_mount(cloud, config, tmp_path):
    class RegionalLambda(FakeLambda):
        def request(self, method, endpoint, body=None):
            if endpoint == "filesystems":
                self.calls.append((method, endpoint, body))
                return [{"id": "old-fs", "name": "scalar-pi"}] if method == "GET" else {"id": "new-fs"}
            return super().request(method, endpoint, body)

    config.update(instance_type="gpu_1x_h100_sxm5", region="us-south-2", filesystem_name="scalar-pi-us-south-2")
    cloud.validate_config(config)
    key = Path(config["ssh_key_path"])
    key.write_text("fixture key")
    key.with_suffix(".pub").write_text("ssh-ed25519 fixture")
    retained = [{"id": "old-fs", "name": "scalar-pi", "region": "us-west-3"}]
    state = {"runs": [], "resources": {"ssh_key_id": "key", "retained_filesystems": retained}}
    client = RegionalLambda(config)
    cloud.prepare(client, config, state, tmp_path / "state.json")
    assert state["resources"]["retained_filesystems"] == retained
    assert state["resources"]["filesystem_id"] == "new-fs"
    assert ("POST", "filesystems", {"name": "scalar-pi-us-south-2", "region": "us-south-2"}) in client.calls
    assert not any(endpoint == "instance-operations/launch" for _, endpoint, _ in client.calls)


@pytest.mark.parametrize("stage", ["qualification", "research"])
def test_canary_exception_does_not_authorize_training(cloud, canary_config, tmp_path, stage):
    client = FakeLambda(canary_config)
    args = SimpleNamespace(stage=stage)
    with pytest.raises(ValueError, match="canary-only"):
        cloud.run_session(client, canary_config, {"runs": []}, tmp_path / "state.json", args)
    assert client.calls == []


def test_canary_exception_rejects_arbitrary_worker_script(cloud, canary_config, tmp_path):
    client = FakeLambda(canary_config)
    args = SimpleNamespace(stage="canary", script=ROOT / "scripts/lambda/train.sh")
    with pytest.raises(ValueError, match="canonical"):
        cloud.run_session(client, canary_config, {"runs": []}, tmp_path / "state.json", args)
    assert client.calls == []
    canary_config["canary_allowance_usd"] = 5.01
    with pytest.raises(ValueError, match=r"exceed \$5"):
        cloud.validate_config(canary_config)


def test_unverified_prepare_checks_canary_budget_before_resource_creation(cloud, canary_config, tmp_path):
    canary_config["canary_allowance_usd"] = 1
    client = FakeLambda(canary_config)
    with pytest.raises(ValueError, match="Cumulative canary"):
        cloud.prepare(client, canary_config, {"resources": {}, "runs": []}, tmp_path / "state.json")
    assert not Path(canary_config["ssh_key_path"]).exists()
    assert [method for method, _, _ in client.calls] == ["GET"]


@pytest.mark.parametrize("profile", [("gpu_1x_h100_pcie", "us-west-3"), ("gpu_1x_h100_sxm5", "us-south-2")])
def test_approved_canary_launches_and_terminates_without_claiming_verified_billing(
    cloud, canary_config, tmp_path, monkeypatch, profile
):
    canary_config.update(instance_type=profile[0], region=profile[1], canary_allowance_usd=5)
    cloud.validate_config(canary_config)
    client = FakeLambda(canary_config)
    state = {"schema_version": 1, "resources": {"ssh_key_id": "key", "filesystem_id": "fs"}, "runs": []}
    args = SimpleNamespace(
        max_hours=0.25,
        stage="canary",
        budget_report=None,
        commit="a" * 40,
        script=ROOT / "scripts/lambda/canary.sh",
    )
    monkeypatch.setattr(cloud.subprocess, "run", lambda *_a, **_k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(cloud.subprocess, "Popen", lambda *_a, **_k: (_ for _ in ()).throw(OSError("fixture")))
    with pytest.raises(OSError, match="fixture"):
        cloud.run_session(client, canary_config, state, tmp_path / "state.json", args)
    assert client.status == "terminated"
    assert state["runs"][0]["allowance"]["canary_allowance_usd"] == 5
    launch = next(body for _, endpoint, body in client.calls if endpoint == "instance-operations/launch")
    assert (launch["instance_type_name"], launch["region_name"]) == profile
    assert canary_config["credits_verified"] is False


def test_research_scope_is_locked_before_results(cloud):
    state = {}
    report = {
        "selected_study": "pilot",
        "experiment_identity": {"code_commit": "a" * 40, "container_image_id": "sha256:" + "b" * 64},
        "study_lock_sha256": "c" * 64,
    }
    cloud.bind_study(state, report, "a" * 40)
    cloud.bind_study(state, report, "a" * 40)
    with pytest.raises(ValueError, match="originally admitted"):
        cloud.bind_study(state, {**report, "selected_study": "full"}, "a" * 40)
    with pytest.raises(ValueError, match="receipt"):
        cloud.bind_study({}, {**report, "study_lock_sha256": ""}, "a" * 40)


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
        "forecast_hourly_rate_usd": 3.79,
        "session_hours": 4,
        "startup_seconds_per_session": 10,
        "shutdown_reserve_seconds": 600,
        "storage": {
            "gib_month_rate_usd": 0.2,
            "current_gib": 0,
            "additional_gib": 0,
            "retention_hours": 720,
            "measurement_source": "test fixture",
        },
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
