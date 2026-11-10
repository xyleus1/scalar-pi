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


def probe(root, destination, boot_id):
    report = {"schema_version": 1, "passed": False, "started_unix": time.time(), "boot_id": boot_id}
    try:
        report["filesystem_type"] = command("findmnt", "-T", str(root), "-n", "-o", "FSTYPE")
        if report["filesystem_type"] not in ("nfs", "nfs4"):
            raise RuntimeError("Expected the attached persistent Lambda filesystem")
        report["gpu"] = command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")
        if len(report["gpu"].splitlines()) != 1 or "H100" not in report["gpu"]:
            raise RuntimeError("Expected exactly one H100")
        report["docker_version"] = command("sudo", "-n", "docker", "version", "--format", "{{.Server.Version}}")
        report["docker_runtimes"] = command("sudo", "-n", "docker", "info", "--format", "{{json .Runtimes}}")
        if "nvidia" not in json.loads(report["docker_runtimes"]):
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
    parser.add_argument("--root", type=Path, default=Path("/lambda/nfs/scalar-pi"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if args.action == "verify":
        print(json.dumps(verify_retained(args.root, boot_id)))
        return 0
    if args.report is None or args.report.exists():
        parser.error("probe requires a new --report path")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    return probe(args.root, args.report, boot_id)


if __name__ == "__main__":
    raise SystemExit(main())
