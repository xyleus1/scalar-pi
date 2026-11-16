"""Infrastructure-only probe and retained storage verification; no ML imports."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def verify_retained(root, boot_id):
    directory = root / "data/experiments/reports/canary"
    receipts = sorted(directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt.get("passed") is not True:
            continue
        marker = Path(receipt["marker_path"]).resolve()
        if marker.parent != directory.resolve() or marker != path.with_suffix(".marker").resolve():
            raise ValueError("Canary marker must belong to its retained receipt")
        if hashlib.sha256(marker.read_bytes()).hexdigest() != receipt["marker_sha256"]:
            raise ValueError("Retained canary marker is corrupt")
        if receipt["boot_id"] == boot_id:
            raise ValueError("Persistence must be verified on a later instance boot")
        return {"receipt": str(path), "marker_sha256": receipt["marker_sha256"], "verified_boot_id": boot_id}
    raise ValueError("Complete the infrastructure canary before staging")


def command(*argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def validate_mount(root, mounts):
    root = root.resolve()
    if root == Path("/") or len(mounts) != 1:
        raise RuntimeError("Expected one distinct mount for the attached Lambda filesystem")
    mount = mounts[0]
    target = mount.get("target")
    if not target or Path(target).resolve() != root:
        raise RuntimeError("Lambda filesystem mount target does not match the requested root")
    if not mount.get("source") or mount.get("fstype") not in ("nfs", "nfs4", "virtiofs"):
        raise RuntimeError("Expected the attached persistent Lambda filesystem")
    return mount


def verify_project(root, storage_root, boot_id):
    """Verify one study directory against the retained physical filesystem."""
    root, storage_root = Path(root), Path(storage_root)
    for path in (root, storage_root):
        if not path.is_absolute() or not path.is_dir() or path.resolve() != path or path.is_symlink():
            raise ValueError("Study and storage roots must be existing canonical directories")
    if not root.is_relative_to(storage_root):
        raise ValueError("Study root must remain inside the physical storage root")
    mount = validate_mount(
        storage_root,
        json.loads(
            command("findmnt", "--json", "--mountpoint", str(storage_root), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS")
        )["filesystems"],
    )
    actual = json.loads(
        command("findmnt", "--json", "--target", str(root), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS")
    )["filesystems"]
    identity = ("target", "source", "fstype")
    if len(actual) != 1 or any(actual[0].get(key) != mount[key] for key in identity):
        raise ValueError("Study root does not resolve to the verified physical filesystem")
    retained = verify_retained(storage_root, boot_id)
    receipt = json.loads(Path(retained["receipt"]).read_text())
    original = validate_mount(storage_root, receipt["filesystem_mounts"])
    if any(original.get(key) != mount[key] for key in identity):
        raise ValueError("Actual storage differs from the retained persistence receipt")
    return {
        **retained,
        "study_root": str(root),
        "storage_root": str(storage_root),
        "filesystem": {key: mount[key] for key in identity},
    }


def probe(root, destination, boot_id):
    report = {"schema_version": 1, "passed": False, "started_unix": time.time(), "boot_id": boot_id}
    try:
        # --mountpoint does not fall back to an ancestor (including the guest root).
        report["filesystem_mounts"] = json.loads(
            command(
                "findmnt", "--json", "--mountpoint", str(root.resolve()), "--output", "TARGET,SOURCE,FSTYPE,OPTIONS"
            )
        )["filesystems"]
        mount = validate_mount(root, report["filesystem_mounts"])
        report["filesystem_type"] = mount["fstype"]
        report["gpu"] = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
        if len(report["gpu"].splitlines()) != 1 or "H100" not in report["gpu"]:
            raise RuntimeError("Expected exactly one H100")
        report["docker_version"] = command("sudo", "-n", "docker", "version", "--format", "{{.Server.Version}}")
        report["docker_runtimes"] = sorted(
            json.loads(command("sudo", "-n", "docker", "info", "--format", "{{json .Runtimes}}"))
        )
        if "nvidia" not in report["docker_runtimes"]:
            raise RuntimeError("Docker NVIDIA runtime is unavailable")
        memory_kib = int(
            next(
                line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:")
            ).split()[1]
        )
        report["host_memory_bytes"] = memory_kib * 1024
        report["local_free_bytes"] = shutil.disk_usage("/").free
        if memory_kib < 64 * 1024 * 1024 or report["local_free_bytes"] < 100 * 1024**3:
            raise RuntimeError("Insufficient host RAM or local staging space")
        marker = destination.with_suffix(".marker")
        with marker.open("xb") as stream:
            stream.write(b"scalar-pi persistent filesystem canary\n")
            stream.flush()
            os.fsync(stream.fileno())
        report["marker_path"] = str(marker)
        report["marker_sha256"] = hashlib.sha256(marker.read_bytes()).hexdigest()
        report["passed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["finished_unix"] = time.time()
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "verify"))
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("SCALAR_PI_ROOT", "/lambda/nfs/scalar-pi")))
    parser.add_argument(
        "--storage-root", type=Path, default=Path(os.environ.get("SCALAR_PI_STORAGE_ROOT", "/lambda/nfs/scalar-pi"))
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if args.action == "verify":
        print(json.dumps(verify_project(args.root, args.storage_root, boot_id)))
        return 0
    if args.report is None or args.report.exists():
        parser.error("probe requires a new --report path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    return probe(args.root, args.report, boot_id)


if __name__ == "__main__":
    raise SystemExit(main())
