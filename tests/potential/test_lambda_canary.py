"""Retained storage verification uses actual fixture files without cloud calls."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest

CANARY = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/lambda/canary.py"))
VERIFY = CANARY["verify_retained"]


@pytest.mark.parametrize("fstype", ["nfs", "nfs4", "virtiofs"])
def test_canary_accepts_exact_shared_filesystem_mount(tmp_path, fstype):
    mount = {"target": str(tmp_path), "source": "provider-share", "fstype": fstype, "options": "rw"}
    assert CANARY["validate_mount"](tmp_path, [mount]) == mount


@pytest.mark.parametrize("target", ["/", "/lambda/nfs"])
def test_canary_rejects_ancestor_mount_and_preserves_evidence(tmp_path, monkeypatch, target):
    root = Path("/lambda/nfs/scalar-pi")
    mounts = [{"target": target, "source": "provider-share", "fstype": "virtiofs", "options": "rw"}]

    def fake_command(*argv):
        assert argv == (
            "findmnt",
            "--json",
            "--mountpoint",
            str(root.resolve()),
            "--output",
            "TARGET,SOURCE,FSTYPE,OPTIONS",
        )
        return json.dumps({"filesystems": mounts})

    probe = CANARY["probe"]
    monkeypatch.setitem(probe.__globals__, "command", fake_command)
    destination = tmp_path / "failed.json"
    assert probe(root, destination, "boot") == 1
    report = json.loads(destination.read_text())
    assert report["passed"] is False
    assert report["filesystem_mounts"] == mounts
    assert "mount target" in report["error"]["message"]
    assert not destination.with_suffix(".marker").exists()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "empty-source", "local-filesystem"])
def test_canary_rejects_missing_or_ambiguous_shared_mount(tmp_path, failure):
    mount = {"target": str(tmp_path), "source": "provider-share", "fstype": "virtiofs"}
    mounts = [mount]
    if failure == "missing":
        mounts = []
    elif failure == "duplicate":
        mounts = [mount, mount.copy()]
    elif failure == "empty-source":
        mount["source"] = ""
    else:
        mount["fstype"] = "ext4"
    with pytest.raises(RuntimeError, match="Lambda filesystem"):
        CANARY["validate_mount"](tmp_path, mounts)


def test_canary_requires_retained_bytes_on_a_different_boot(tmp_path):
    with pytest.raises(ValueError, match="before staging"):
        VERIFY(tmp_path, "new-boot")
    directory = tmp_path / "data/experiments/reports/canary"
    directory.mkdir(parents=True)
    marker = directory / "test.marker"
    marker.write_bytes(b"retained evidence")
    receipt = {
        "passed": True,
        "marker_path": str(marker),
        "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
        "boot_id": "old-boot",
    }
    (directory / "test.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="later instance"):
        VERIFY(tmp_path, "old-boot")
    assert VERIFY(tmp_path, "new-boot")["marker_sha256"] == receipt["marker_sha256"]
    marker.write_bytes(b"damaged")
    with pytest.raises(ValueError, match="corrupt"):
        VERIFY(tmp_path, "new-boot")
