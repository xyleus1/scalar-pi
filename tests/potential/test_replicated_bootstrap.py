"""Offline local-storage admission tests; no mount, cloud or SSH is used."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

FILE = Path(__file__).resolve().parents[2] / "scripts/lambda/replicated_bootstrap.py"
ATTEMPT = "scalar-pi-012345abcdef"
SHA = "a" * 64


@pytest.fixture
def module():
    spec = importlib.util.spec_from_file_location("replicated_storage_test", FILE)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def host(module, tmp_path, monkeypatch):
    for key, leaf in (("SOURCE", "local/source"), ("TARGET", "mount/target"), ("EVIDENCE", "host/evidence")):
        monkeypatch.setattr(module, key, tmp_path / leaf)
    module.SOURCE.parent.mkdir()
    state = SimpleNamespace(mounted=False, calls=[], ownership=[], fstype="ext4", inode_matches=True)
    monkeypatch.setattr(module.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(module.os, "chown", lambda *args: state.ownership.append(args), raising=False)
    monkeypatch.setattr(module, "ubuntu", lambda: SimpleNamespace(pw_uid=0, pw_gid=0))
    monkeypatch.setattr(module, "host_identity", lambda: {"machine_id": "a" * 32, "boot_id": "b" * 36, "ram_bytes": 4096})
    monkeypatch.setattr(module, "free_bytes", lambda path: 123456)
    real_identity = module.identity

    def identity(path):
        if path == module.TARGET and state.mounted and state.inode_matches:
            return real_identity(module.SOURCE)
        return real_identity(path)

    monkeypatch.setattr(module, "identity", identity)

    def run(argv, deadline, *, now):
        module.remaining(deadline, now)
        state.calls.append(argv)
        if argv[:2] == ["mount", "--bind"]:
            state.mounted = True
            return SimpleNamespace(returncode=0, stdout="")
        assert argv[0] == "findmnt"
        if "--mountpoint" in argv and not state.mounted:
            return SimpleNamespace(returncode=1, stdout="")
        row = {"target": str(module.TARGET) if "--mountpoint" in argv else "/",
               "source": "/dev/nvme0n1p1[/var/lib/scalar-pi-replica]", "fstype": state.fstype,
               "options": "rw,relatime", "maj:min": "259:1"}
        return SimpleNamespace(returncode=0, stdout=json.dumps({"filesystems": [row]}))

    state.run = run
    state.bootstrap = lambda: module.bootstrap(ATTEMPT, 4000, SHA, run=run, now=lambda: 1000)
    return state


def test_actual_bind_is_ephemeral_and_records_verified_local_identity(module, host):
    value = host.bootstrap()
    assert value["status"] == "bound"
    assert value["storage_mode"] == "replicated_local" and value["persistence"] == "ephemeral"
    assert value["qualification"] is False and value["free_bytes"] == 123456
    assert value["host"]["ram_bytes"] == 4096
    assert value["directory_identity"] == module.identity(module.TARGET)
    assert json.loads((module.EVIDENCE / "storage-binding.json").read_text()) == value
    assert [v for v in host.calls if v[0] == "mount"] == [["mount", "--bind", str(module.SOURCE), str(module.TARGET)]]
    assert host.ownership == [(module.SOURCE, 0, 0)]


def test_exact_completed_replay_only_rechecks_current_mount(module, host):
    expected = host.bootstrap()
    host.calls.clear()
    assert host.bootstrap() == expected
    assert all(v[0] == "findmnt" for v in host.calls)


@pytest.mark.parametrize("conflict", ["source", "nonempty_target", "partial_evidence", "foreign_mount"])
def test_unknown_existing_storage_never_gets_replaced(module, host, conflict):
    if conflict == "source":
        module.SOURCE.mkdir()
    elif conflict == "nonempty_target":
        module.TARGET.mkdir(parents=True)
        (module.TARGET / "keep").write_text("actual data")
    elif conflict == "partial_evidence":
        module.EVIDENCE.mkdir(parents=True)
    else:
        host.mounted = True
    with pytest.raises(ValueError):
        host.bootstrap()
    assert not any(v[0] == "mount" for v in host.calls)
    assert not (module.EVIDENCE / "storage-binding.json").exists()


@pytest.mark.parametrize("filesystem", ["virtiofs", "nfs4", "tmpfs", "overlay"])
def test_only_actual_local_disk_is_admitted(module, host, filesystem):
    host.fstype = filesystem
    with pytest.raises(ValueError, match="local disk"):
        host.bootstrap()
    assert not module.EVIDENCE.exists()


def test_same_device_without_same_inode_cannot_claim_a_bind(module, host):
    host.inode_matches = False
    with pytest.raises(ValueError, match="device or inode"):
        host.bootstrap()
    assert (module.EVIDENCE / "installation-intent.json").exists()
    assert not (module.EVIDENCE / "storage-binding.json").exists()
    before = len(host.calls)
    with pytest.raises(ValueError, match="reconciliation"):
        host.bootstrap()
    assert len(host.calls) == before


def test_lost_mount_response_is_never_retried(module, host):
    def uncertain(argv, deadline, *, now):
        result = host.run(argv, deadline, now=now)
        if argv[0] == "mount":
            raise subprocess.TimeoutExpired(argv, 1)
        return result

    with pytest.raises(subprocess.TimeoutExpired):
        module.bootstrap(ATTEMPT, 4000, SHA, run=uncertain, now=lambda: 1000)
    before = len(host.calls)
    with pytest.raises(ValueError, match="reconciliation"):
        host.bootstrap()
    assert len(host.calls) == before


def test_mount_query_error_is_not_absence(module, host):
    def failed(argv, deadline, *, now):
        return SimpleNamespace(returncode=2, stdout="")

    with pytest.raises(ValueError, match="mount identity"):
        module.bootstrap(ATTEMPT, 4000, SHA, run=failed, now=lambda: 1000)
    assert not module.EVIDENCE.exists()


def test_completed_receipt_cannot_be_reused_by_another_attempt(module, host):
    host.bootstrap()
    with pytest.raises(ValueError, match="another installation"):
        module.bootstrap("scalar-pi-ffffffffffff", 4000, SHA, run=host.run, now=lambda: 1000)


@pytest.mark.parametrize("deadline", [1015, 15401, float("inf"), float("nan"), True])
def test_invalid_or_late_deadline_changes_nothing(module, host, deadline):
    with pytest.raises(ValueError):
        module.bootstrap(ATTEMPT, deadline, SHA, run=host.run, now=lambda: 1000)
    assert not host.calls and not module.EVIDENCE.exists()


def test_subprocess_timeout_is_bounded_by_actual_deadline(module, monkeypatch):
    captured = []
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: captured.append(kwargs))
    module.command(["findmnt"], 1020, now=lambda: 1000)
    assert captured[0]["timeout"] == 5
    with pytest.raises(TimeoutError):
        module.command(["mount"], 1015, now=lambda: 1000)
    assert len(captured) == 1


def test_cloud_init_source_and_actual_cli_contract_preview(module, tmp_path):
    argv = module.cloud_init_command(ATTEMPT, 4000)
    assert argv[:2] == ["python3", "-c"]
    assert argv[2].encode() == FILE.read_bytes()
    assert argv[argv.index("--helper-sha256") + 1] == hashlib.sha256(FILE.read_bytes()).hexdigest()
    # Execute the actual embedded source with execute omitted. It must not
    # attempt root/mount/filesystem operations, even on Windows.
    result = subprocess.run([sys.executable, *argv[1:-1]], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "preview_no_calls"
    assert list(tmp_path.iterdir()) == []
