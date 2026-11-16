"""Project subroots retain physical persistence evidence without relabeling it."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("namespace_canary", REPO / "scripts/lambda/canary.py")
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    project = root / "protocols/hybrid-v1"
    project.mkdir(parents=True)
    directory = root / "data/experiments/reports/canary"
    directory.mkdir(parents=True)
    marker = directory / "old.marker"
    marker.write_bytes(b"actual retained fixture bytes")
    mount = {"target": str(root), "source": "fixture-virtiofs", "fstype": "virtiofs", "options": "rw"}
    receipt = directory / "old.json"
    receipt.write_text(
        json.dumps(
            {
                "passed": True,
                "boot_id": "prior-boot",
                "marker_path": str(marker),
                "marker_sha256": hashlib.sha256(marker.read_bytes()).hexdigest(),
                "filesystem_mounts": [mount],
            }
        )
    )
    calls = []

    def command(*argv):
        calls.append(argv)
        assert argv[0] == "findmnt"
        assert "--mountpoint" in argv or "--target" in argv
        return json.dumps({"filesystems": [mount]})

    monkeypatch.setattr(canary, "command", command)
    return root, project, receipt, mount, calls


def test_subroot_reuses_original_marker_and_boot_evidence_without_copy(storage):
    root, project, receipt, mount, calls = storage
    original = receipt.read_bytes()
    result = canary.verify_project(project, root, "new-boot")
    assert result["receipt"] == str(receipt)
    assert result["study_root"] == str(project)
    assert result["filesystem"] == {key: mount[key] for key in ("target", "source", "fstype")}
    assert receipt.read_bytes() == original
    assert not (project / "data/experiments/reports/canary").exists()
    assert calls[0][2:4] == ("--mountpoint", str(root))
    assert calls[1][2:4] == ("--target", str(project))


def test_original_project_root_and_same_boot_gate_remain(storage):
    root, _, receipt, _, _ = storage
    assert canary.verify_project(root, root, "new-boot")["receipt"] == str(receipt)
    with pytest.raises(ValueError, match="later instance boot"):
        canary.verify_project(root, root, "prior-boot")


@pytest.mark.parametrize(
    "damage", ["outside", "noncanonical", "missing", "wrong_target", "wrong_source", "retained_source"]
)
def test_subroot_escape_or_different_filesystem_is_rejected(storage, monkeypatch, damage):
    root, project, receipt, mount, _ = storage
    if damage == "outside":
        project = root.parent
    elif damage == "noncanonical":
        project = root / "protocols/../protocols/hybrid-v1"
    elif damage == "missing":
        project = root / "protocols/absent"
    elif damage == "retained_source":
        value = json.loads(receipt.read_text())
        value["filesystem_mounts"][0]["source"] = "different-filesystem"
        receipt.write_text(json.dumps(value))
    else:

        def command(*argv):
            current = dict(mount)
            if "--target" in argv:
                current["target" if damage == "wrong_target" else "source"] = "different"
            return json.dumps({"filesystems": [current]})

        monkeypatch.setattr(canary, "command", command)
    with pytest.raises((ValueError, RuntimeError)):
        canary.verify_project(project, root, "new-boot")


def test_matrix_cli_uses_environment_project_root(tmp_path, monkeypatch):
    root = tmp_path.resolve() / "protocols/hybrid-v1"
    output = tmp_path.resolve() / "diagnostic.json"
    monkeypatch.setenv("SCALAR_PI_ROOT", root.as_posix())
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/lambda/matrix.py"),
            "render",
            "--diagnostic",
            "--attempt",
            "namespace-test",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    manifest = json.loads(output.read_text())
    assert Path(manifest["output_dir"]).is_relative_to(root)
    assert all(Path(v["checkpoint_dir"]).is_relative_to(root) for v in manifest["variants"])
    assert manifest["diagnostic"] is True
    assert manifest["num_trials_per_task"] == 1


def test_calibration_default_receives_exact_environment_root(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts/lambda"))
    module_spec = importlib.util.spec_from_file_location(
        "namespace_calibration_spec", REPO / "scripts/lambda/calibration_spec.py"
    )
    calibration_spec = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(calibration_spec)
    seen = []
    monkeypatch.setenv("SCALAR_PI_ROOT", "/lambda/nfs/scalar-pi/protocols/hybrid-v1")
    monkeypatch.setenv("SCALAR_PI_ATTEMPT", "namespace-fixture")
    monkeypatch.setattr(calibration_spec, "prepare", lambda root, **kwargs: seen.append((root, kwargs)))
    calibration_spec.main([])
    assert seen == [
        (Path("/lambda/nfs/scalar-pi/protocols/hybrid-v1"), {"attempt": "namespace-fixture", "smoke_manifest": None})
    ]


def test_shell_default_paths_and_container_environment_stay_consistent():
    for name in ("prepare.sh", "profile.sh", "train.sh", "calibrate.sh", "smoke_libero.sh", "run.sh"):
        source = (REPO / "scripts/lambda" / name).read_text()
        assert 'root="${SCALAR_PI_ROOT:-/lambda/nfs/scalar-pi}"' in source
        assert 'export SCALAR_PI_ROOT="$root"' in source
    source = (REPO / "scripts/lambda/run.sh").read_text()
    assert 'canary.py verify --root "$root" --storage-root "$storage_root"' in source
    assert '-e "SCALAR_PI_ROOT=$root" -e "SCALAR_PI_STORAGE_ROOT=$storage_root"' in source
    assert 'root="${SCALAR_PI_STORAGE_ROOT:-/lambda/nfs/scalar-pi}"' in (REPO / "scripts/lambda/canary.sh").read_text()
    # Billing/instance cleanup and ceaf infrastructure stay at the physical root.
    assert 'PERSISTENT_ROOT = Path("/lambda/nfs/scalar-pi")' in (REPO / "scripts/lambda/detached.py").read_text()
    assert 'REMOTE_ROOT = "/lambda/nfs/scalar-pi"' in (REPO / "scripts/lambda/cloud.py").read_text()
