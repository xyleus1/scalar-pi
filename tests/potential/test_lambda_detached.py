"""Offline service/ownership tests; no systemd service, SSH or cloud is used."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = "scalar-pi-012345abcdef"


@pytest.fixture
def detached():
    spec = importlib.util.spec_from_file_location("scalar_pi_detached_test", ROOT / "scripts/lambda/detached.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def installation(detached, tmp_path, monkeypatch):
    monkeypatch.setattr(detached, "LOCAL_ROOT", tmp_path / "host" / "workers")
    monkeypatch.setattr(detached, "PERSISTENT_ROOT", tmp_path / "shared")
    monkeypatch.setattr(detached.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(detached.os, "chown", lambda *args: None, raising=False)
    monkeypatch.setattr(detached, "ubuntu", lambda: SimpleNamespace(pw_uid=0, pw_gid=0))
    monkeypatch.setattr(detached, "mounted", lambda: True)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="loaded\n")

    monkeypatch.setattr(detached.subprocess, "run", run)
    payload = tmp_path / "input.sh"
    payload.write_text("set -euo pipefail\nexit 42\n")
    deadline = detached.time.time() + 3600
    detached.install(ATTEMPT, deadline, payload)
    return SimpleNamespace(module=detached, payload=payload, deadline=deadline, calls=calls, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    "attempt", ["scalar-pi-../other", "scalar-pi-x;touch_bad", "other-012345abcdef", "scalar-pi-ABCDEF123456"]
)
def test_attempt_names_cannot_select_other_paths_or_units(detached, attempt):
    with pytest.raises(ValueError, match="attempt name"):
        detached.build_remote_script("true\n", attempt=attempt, deadline_unix=100)


@pytest.mark.parametrize("deadline", [True, 0, -1, float("nan"), float("inf"), "123"])
def test_invalid_deadline_rejected(detached, deadline):
    with pytest.raises(ValueError, match="Deadline"):
        detached.build_remote_script("true\n", attempt=ATTEMPT, deadline_unix=deadline)


def test_generated_shell_parses_and_transports_payload_verbatim(detached, tmp_path):
    payload = "set -euo pipefail\nprintf '%s' '$HOME `not-code`'\nexit 42\n"
    wrapper = detached.build_remote_script(payload, attempt=ATTEMPT, deadline_unix=2000000000.5)
    encoded = re.findall(r"printf '%s' '([A-Za-z0-9+/=]+)'", wrapper)
    assert len(encoded) == 2
    assert base64.b64decode(encoded[1]).decode() == payload
    assert "sudo -n -u ubuntu" in wrapper
    assert f"/var/lib/scalar-pi/workers/{ATTEMPT}/detached.py monitor" in wrapper
    assert "trap" in wrapper
    assert "systemctl stop" not in wrapper
    path = tmp_path / "wrapper.sh"
    path.write_text(wrapper, encoding="utf-8", newline="\n")
    bash = Path("C:/Program Files/Git/bin/bash.exe") if os.name == "nt" else shutil.which("bash")
    if not bash or not Path(bash).is_file():
        pytest.skip("Bash is unavailable for the offline syntax check")
    subprocess.run([str(bash), "-n", str(path)], check=True, capture_output=True, timeout=15)


def test_cloud_key_is_never_accepted_in_worker_program(detached):
    with pytest.raises(ValueError, match="credentials"):
        detached.build_remote_script("export LAMBDA_API_KEY=fixture\n", attempt=ATTEMPT, deadline_unix=100)


def test_cloud_init_bootstrap_does_not_wait_for_worker(detached):
    wrapper = detached.build_remote_script(
        "sudo cloud-init status --wait\ntrue\n", attempt=ATTEMPT, deadline_unix=2000000000.5, bootstrap_only=True
    )
    assert "install --attempt" in wrapper
    assert " monitor --attempt " not in wrapper
    assert "--wait" not in wrapper  # The payload is encoded, never run by cloud-init itself.
    with pytest.raises(ValueError, match="boolean"):
        detached.build_remote_script("true\n", attempt=ATTEMPT, deadline_unix=100, bootstrap_only="yes")


def test_standard_service_is_owned_by_ubuntu_and_bounded(detached):
    args = detached.service_command(ATTEMPT, 4600, Path("/var/lib/scalar-pi/workers/x/detached.py"), now=1000)
    assert args[0] == "systemd-run"
    assert "--property=Type=exec" in args
    assert {"--scope", "--pipe", "--pty", "--wait", "--user"}.isdisjoint(args)
    assert "--uid=ubuntu" in args
    assert "--gid=ubuntu" in args
    assert "--property=Restart=no" in args
    assert "--property=RuntimeMaxSec=3585s" in args
    assert "--property=TimeoutStopSec=10s" in args
    assert any(arg.startswith("--property=ExecStopPost=") for arg in args)
    assert "--property=StandardInput=null" in args
    with pytest.raises(ValueError, match="bounded lease"):
        detached.service_command(ATTEMPT, 1015, Path("/worker.py"), now=1000)


def test_nonroot_install_is_rejected_before_filesystem_writes(detached, tmp_path, monkeypatch):
    monkeypatch.setattr(detached.os, "geteuid", lambda: 1000, raising=False)
    with pytest.raises(PermissionError):
        detached.install(ATTEMPT, detached.time.time() + 100, tmp_path / "missing.sh")


def test_reconnection_never_starts_another_service(installation):
    item = installation
    item.module.install(ATTEMPT, item.deadline, item.payload)
    assert sum(command[0] == "systemd-run" for command in item.calls) == 1
    assert item.calls[-1][0] == "systemctl"


def test_same_attempt_with_changed_payload_is_not_reused(installation):
    item = installation
    item.payload.write_text("exit 0\n")
    with pytest.raises(ValueError, match="different"):
        item.module.install(ATTEMPT, item.deadline, item.payload)
    assert sum(command[0] == "systemd-run" for command in item.calls) == 1


def test_partial_installation_is_not_automatically_restarted(installation):
    item = installation
    item.monkeypatch.setattr(
        item.module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="not-found\n")
    )
    with pytest.raises(ValueError, match="automatic restart"):
        item.module.install(ATTEMPT, item.deadline, item.payload)


def test_worker_refuses_root_when_ubuntu_has_another_uid(installation):
    item = installation
    item.monkeypatch.setattr(item.module, "ubuntu", lambda: SimpleNamespace(pw_uid=1000, pw_gid=1000))
    with pytest.raises(PermissionError, match="ubuntu"):
        item.module.worker(ATTEMPT)


def test_unmounted_shared_root_is_never_created(detached, tmp_path, monkeypatch):
    monkeypatch.setattr(detached, "PERSISTENT_ROOT", tmp_path / "not-mounted")
    monkeypatch.setattr(detached, "mounted", lambda: False)
    assert detached.persistent_directory(ATTEMPT) is None
    assert not detached.PERSISTENT_ROOT.exists()


def test_payload_exit_and_logs_are_persistent_and_monitor_only_observes(installation):
    item = installation
    commands = []

    def run(command, output, deadline):
        assert deadline == item.deadline
        commands.append(command)
        output.write(b"fixture output\n")
        return 42 if command[0] == "/bin/bash" else 0

    item.monkeypatch.setattr(item.module, "run_command", run)
    assert item.module.worker(ATTEMPT) == 42
    assert commands[0] == ["sudo", "-n", "cloud-init", "status", "--wait"]
    output = item.module.PERSISTENT_ROOT / "data/operations/detached" / ATTEMPT
    result = json.loads((output / "result.json").read_text())
    assert result["exit_code"] == 42
    assert result["status"] == "payload_failed"
    assert result["deadline_unix"] == item.deadline
    assert (output / "worker.log").read_bytes() == b"fixture output\n"
    service_calls = len(item.calls)
    assert item.module.monitor(ATTEMPT) == 42
    assert len(item.calls) == service_calls
    with pytest.raises(ValueError, match="only once"):
        item.module.worker(ATTEMPT)
    assert len(commands) == 2


def test_forced_service_exit_has_fallback_receipt(installation):
    item = installation
    item.monkeypatch.setenv("SERVICE_RESULT", "exit-code")
    item.monkeypatch.setenv("EXIT_CODE", "exited")
    item.monkeypatch.setenv("EXIT_STATUS", "37")
    item.module.finalize(ATTEMPT)
    output = item.module.PERSISTENT_ROOT / "data/operations/detached" / ATTEMPT
    result = json.loads((output / "result.json").read_text())
    assert result["exit_code"] == 37
    assert result["status"] == "service_ended_without_worker_receipt"
    assert json.loads((output / "service.json").read_text())["systemd"]["EXIT_STATUS"] == "37"


def test_installed_script_tampering_is_rejected(installation):
    item = installation
    (item.module.LOCAL_ROOT / ATTEMPT / "payload.sh").write_text("exit 0\n")
    with pytest.raises(ValueError, match="payload changed"):
        item.module.worker(ATTEMPT)
