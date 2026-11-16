"""Bind a fresh local replica directory for a bounded, ephemeral worker.

This prepares storage only. It downloads nothing and makes no cloud calls. A
partial installation is retained for explicit reconciliation, never repeated.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

SOURCE = Path("/var/lib/scalar-pi-replica")
TARGET = Path("/lambda/nfs/scalar-pi")
EVIDENCE = Path("/var/lib/scalar-pi/replicated-storage")
LOCAL_FILESYSTEMS = {"ext4", "xfs", "btrfs"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(attempt, deadline):
    require(isinstance(attempt, str) and re.fullmatch(r"scalar-pi-[a-f0-9]{12}", attempt), "Invalid attempt")
    require(type(deadline) in (int, float) and math.isfinite(deadline) and deadline > 0, "Invalid deadline")


def cloud_init_command(attempt, deadline_unix):
    """Return one shell-free cloud-init runcmd entry with exact helper bytes."""
    validate(attempt, deadline_unix)
    source = Path(__file__).read_bytes()
    return ["python3", "-c", source.decode("utf-8"), "--attempt", attempt,
            "--deadline-unix", repr(deadline_unix), "--helper-sha256", hashlib.sha256(source).hexdigest(),
            "--execute"]


def remaining(deadline, now=time.time):
    seconds = deadline - now() - 15
    if seconds <= 0:
        raise TimeoutError("Storage bootstrap reached its original stop window")
    return min(30, seconds)


def command(argv, deadline, *, now=time.time):
    return subprocess.run(argv, capture_output=True, text=True, check=False, timeout=remaining(deadline, now))


def mount_info(path, deadline, *, exact=False, run=command, now=time.time):
    result = run(["findmnt", "--json", "--mountpoint" if exact else "--target", str(path),
                  "--output", "TARGET,SOURCE,FSTYPE,OPTIONS,MAJ:MIN"], deadline, now=now)
    if result.returncode == 1 and exact and not result.stdout.strip():
        return None
    require(result.returncode == 0, "Could not establish actual mount identity")
    rows = json.loads(result.stdout).get("filesystems")
    require(isinstance(rows, list) and len(rows) == 1 and not rows[0].get("children"), "Ambiguous mount identity")
    row = rows[0]
    require(all(isinstance(row.get(key), str) and row[key] for key in ("target", "source", "fstype", "options", "maj:min")),
            "Incomplete mount identity")
    if exact:
        require(row["target"] == str(path), "Unexpected target mount")
    return row


def directory(path, *, create=False):
    """Create only the requested parent chain, without changing existing modes."""
    require(path.is_absolute() and path.resolve() == path, "Use a canonical absolute directory")
    for item in [*reversed(path.parents), path]:
        require(not item.is_symlink(), "Storage directory has a symlink ancestor")
        if not item.exists():
            require(create, "Storage parent is missing")
            item.mkdir(mode=0o755)
        require(item.is_dir() and item.stat().st_uid == 0, "Storage parents must be root-owned directories")


def identity(path):
    value = path.stat()
    return {"device": value.st_dev, "inode": value.st_ino}


def ubuntu():
    import pwd

    return pwd.getpwnam("ubuntu")


def host_identity():
    machine = Path("/etc/machine-id").read_text().strip()
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    require(re.fullmatch(r"[a-f0-9]{32}", machine) and re.fullmatch(r"[a-f0-9-]{36}", boot), "Invalid actual host identity")
    memory = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemTotal:"))
    parts = memory.split()
    require(len(parts) == 3 and parts[2] == "kB", "Unexpected actual RAM units")
    return {"machine_id": machine, "boot_id": boot, "ram_bytes": int(parts[1]) * 1024}


def free_bytes(path):
    value = os.statvfs(path)
    return value.f_bavail * value.f_frsize


def publish(path, value):
    # Exclusive publication: a partial file is evidence, never a retry target.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def verify_mount(binding, deadline, *, run=command, now=time.time):
    account = ubuntu()
    require(not SOURCE.is_symlink() and not TARGET.is_symlink(), "Replica directories cannot be symlinks")
    require(SOURCE.resolve() == SOURCE and TARGET.resolve() == TARGET, "Replica path changed")
    require(SOURCE.stat().st_uid == TARGET.stat().st_uid == account.pw_uid, "Replica ownership differs")
    actual = mount_info(TARGET, deadline, exact=True, run=run, now=now)
    require(actual is not None and actual["fstype"] in LOCAL_FILESYSTEMS, "Expected an actual local filesystem bind")
    require(identity(SOURCE) == identity(TARGET) == binding["directory_identity"], "Bind source/target device or inode differs")
    require(actual == binding["target_mount"], "Actual bind mount changed")
    require(host_identity() == binding["host"], "Actual host or boot changed")
    return binding


def bootstrap(attempt, deadline, helper_sha256, *, run=command, now=time.time):
    validate(attempt, deadline)
    require(re.fullmatch(r"[a-f0-9]{64}", helper_sha256 or ""), "Pin the actual helper bytes")
    require(os.geteuid() == 0, "Storage bootstrap must run as root")
    require(15 < deadline - now() <= 4 * 3600, "Use the actual bounded launch deadline")
    host = host_identity()
    expected = {"schema_version": 1, "attempt": attempt, "deadline_unix": deadline,
                "helper_sha256": helper_sha256, "source": str(SOURCE), "target": str(TARGET),
                "storage_mode": "replicated_local", "persistence": "ephemeral"}
    receipt = EVIDENCE / "storage-binding.json"
    intent = EVIDENCE / "installation-intent.json"
    # No filesystem writes precede these checks.
    if receipt.exists():
        directory(EVIDENCE)
        require(not receipt.is_symlink(), "Binding receipt is a symlink")
        binding = json.loads(receipt.read_text())
        require(all(binding.get(key) == value for key, value in expected.items())
                and binding.get("status") == "bound", "Existing storage binding belongs to another installation")
        return verify_mount(binding, deadline, run=run, now=now)
    require(not EVIDENCE.exists() and not EVIDENCE.is_symlink(), "Partial storage installation requires reconciliation")
    require(not SOURCE.exists() and not SOURCE.is_symlink(), "Replica source already exists without completed evidence")
    if TARGET.exists() or TARGET.is_symlink():
        directory(TARGET)
        require(not any(TARGET.iterdir()), "Replica target must be empty")
    require(mount_info(TARGET, deadline, exact=True, run=run, now=now) is None,
            "Preexisting target mount requires explicit reconciliation")
    # Verify every existing ancestor before creating any requested path.
    for path in (SOURCE, TARGET, EVIDENCE):
        for ancestor in path.parents:
            if ancestor.exists() or ancestor.is_symlink():
                directory(ancestor)
    source_mount = mount_info(SOURCE.parent, deadline, run=run, now=now)
    require(source_mount["fstype"] in LOCAL_FILESYSTEMS and "rw" in source_mount["options"].split(","),
            "Replica source must be on writable local disk")
    account = ubuntu()
    directory(EVIDENCE, create=True)
    publish(intent, {**expected, "host": host, "status": "installation_requested", "observed_unix": now()})
    directory(SOURCE, create=True)
    directory(TARGET, create=True)
    os.chown(SOURCE, account.pw_uid, account.pw_gid)
    result = run(["mount", "--bind", str(SOURCE), str(TARGET)], deadline, now=now)
    require(result.returncode == 0, "Bind outcome requires reconciliation; do not retry the mount")
    target_mount = mount_info(TARGET, deadline, exact=True, run=run, now=now)
    require(target_mount is not None and target_mount["fstype"] == source_mount["fstype"]
            and target_mount["maj:min"] == source_mount["maj:min"], "Bound filesystem differs from the source disk")
    binding = {**expected, "status": "bound", "host": host, "source_mount": source_mount,
               "target_mount": target_mount, "directory_identity": identity(SOURCE),
               "free_bytes": free_bytes(SOURCE), "observed_unix": now(),
               "receipt": str(receipt), "qualification": False}
    verify_mount(binding, deadline, run=run, now=now)
    remaining(deadline, now)
    publish(receipt, binding)
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--deadline-unix", required=True, type=float)
    parser.add_argument("--helper-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    validate(args.attempt, args.deadline_unix)
    if not args.execute:
        print(json.dumps({"status": "preview_no_calls", "storage_mode": "replicated_local", "persistence": "ephemeral"}))
        return
    expected = args.helper_sha256 or hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(bootstrap(args.attempt, args.deadline_unix, expected), sort_keys=True))


if __name__ == "__main__":
    main()
