"""Run a Lambda worker under systemd; SSH only installs and observes it.

The API watchdog still owns cloud-instance termination. No cloud credential is
sent here. A 15-second stop window stays inside the original absolute deadline.
Before the shared filesystem mounts, bootstrap evidence remains in /var/lib;
once mounted, logs and receipts are written to project persistent storage.
"""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import signal
import subprocess
import sys
import time

LOCAL_ROOT = Path("/var/lib/scalar-pi/workers")
PERSISTENT_ROOT = Path("/lambda/nfs/scalar-pi")
STOP_WINDOW = 15
SECRET_ENV = ("LAMBDA_API_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def validate(attempt, deadline_unix):
    if re.fullmatch(r"scalar-pi-[a-f0-9]{12}", attempt or "") is None:
        raise ValueError("Use the exact controller-generated attempt name")
    if isinstance(deadline_unix, bool) or not isinstance(deadline_unix, int | float):
        raise ValueError("Deadline must be a finite Unix timestamp")
    if not math.isfinite(deadline_unix) or deadline_unix <= 0:
        raise ValueError("Deadline must be a finite Unix timestamp")


def unit_name(attempt):
    validate(attempt, 1)
    return f"scalar-pi-worker-{attempt.removeprefix('scalar-pi-')}.service"


def build_remote_script(script, *, attempt, deadline_unix, bootstrap_only=False):
    """Wrap cloud.remote_script's output; cloud-init must use bootstrap_only=True."""
    validate(attempt, deadline_unix)
    if not isinstance(bootstrap_only, bool):
        raise ValueError("bootstrap_only must be a boolean")
    if not isinstance(script, str) or not script.strip() or "\x00" in script:
        raise ValueError("A complete nonempty shell program is required")
    if any(name in script for name in SECRET_ENV):
        raise ValueError("Cloud credentials must stay outside the worker program")
    payload = script.replace("\r\n", "\n").encode()
    helper = Path(__file__).read_bytes()
    destination = PurePosixPath(LOCAL_ROOT.as_posix()) / attempt / "detached.py"
    # Temporary SSH-side files are never service inputs. Installation copies them
    # into a durable host directory before starting the independent service.
    installer = (
        "set -euo pipefail\numask 077\n"
        'transfer=$(mktemp -d)\ntrap \'rm -f -- "$transfer/detached.py" "$transfer/payload.sh"; '
        'rmdir -- "$transfer"\' EXIT\n'
        f"printf '%s' '{base64.b64encode(helper).decode()}' | base64 --decode > \"$transfer/detached.py\"\n"
        f"printf '%s' '{base64.b64encode(payload).decode()}' | base64 --decode > \"$transfer/payload.sh\"\n"
        'sudo -n python3 "$transfer/detached.py" install '
        f"--attempt {shlex.quote(attempt)} --deadline-unix {deadline_unix!r} "
        '--script "$transfer/payload.sh"\n'
    )
    if bootstrap_only:
        return installer
    return installer + (
        f"sudo -n -u ubuntu python3 {shlex.quote(str(destination))} monitor --attempt {shlex.quote(attempt)}\n"
    )


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def ubuntu():
    import pwd

    return pwd.getpwnam("ubuntu")


def remaining(deadline):
    seconds = deadline - STOP_WINDOW - time.time()
    if seconds <= 0:
        raise TimeoutError("Original worker deadline has reached its stop window")
    return seconds


def service_command(attempt, deadline, helper, *, now=None):
    validate(attempt, deadline)
    seconds = math.floor(deadline - (time.time() if now is None else now) - STOP_WINDOW)
    if seconds < 1 or seconds > 4 * 3600:
        raise ValueError("Service must fit the original bounded lease")
    return [
        "systemd-run",
        f"--unit={unit_name(attempt)}",
        "--property=Type=exec",
        "--uid=ubuntu",
        "--gid=ubuntu",
        "--property=Restart=no",
        "--property=KillMode=control-group",
        f"--property=RuntimeMaxSec={seconds}s",
        "--property=TimeoutStartSec=10s",
        "--property=TimeoutStopSec=10s",
        "--property=StandardInput=null",
        "--property=StandardOutput=journal",
        "--property=StandardError=journal",
        f"--property=UnsetEnvironment={' '.join(SECRET_ENV)}",
        f"--property=ExecStopPost=/usr/bin/python3 {helper} finalize --attempt {attempt}",
        "--setenv=HOME=/home/ubuntu",
        "--setenv=USER=ubuntu",
        "--setenv=LOGNAME=ubuntu",
        "--setenv=PYTHONUNBUFFERED=1",
        "/usr/bin/python3",
        str(helper),
        "worker",
        "--attempt",
        attempt,
    ]


def install(attempt, deadline, script):
    validate(attempt, deadline)
    if os.geteuid() != 0:
        raise PermissionError("Only the root installer may create a system service")
    remaining(deadline)
    payload, helper = Path(script).read_bytes(), Path(__file__).read_bytes()
    specification = {
        "schema_version": 1,
        "attempt": attempt,
        "deadline_unix": deadline,
        "stop_window_seconds": STOP_WINDOW,
        "unit": unit_name(attempt),
        "payload_sha256": digest(payload),
        "helper_sha256": digest(helper),
    }
    for parent in (LOCAL_ROOT.parent, LOCAL_ROOT):
        parent.mkdir(mode=0o755, exist_ok=True)
        if parent.is_symlink() or parent.stat().st_uid != 0:
            raise ValueError("Worker installation parents must be root-owned real directories")
        parent.chmod(0o755)
    folder = LOCAL_ROOT / attempt
    if folder.exists() or folder.is_symlink():
        if folder.is_symlink() or json.loads((folder / "request.json").read_text()) != specification:
            raise ValueError("Attempt already exists with different or incomplete installation evidence")
        if not (folder / "state/result.json").is_file():
            status = subprocess.run(
                ["systemctl", "show", specification["unit"], "--property=LoadState", "--value"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            if status.stdout.strip() != "loaded":
                raise ValueError("Partial installation has no service or result; automatic restart is forbidden")
        print(json.dumps({"attempt": attempt, "installation": "already_present", "relaunched": False}), flush=True)
        return
    folder.mkdir(mode=0o755)
    folder.chmod(0o755)
    for name, data in (("detached.py", helper), ("payload.sh", payload)):
        (folder / name).write_bytes(data)
        (folder / name).chmod(0o644)
    write_json(folder / "request.json", specification)
    (folder / "request.json").chmod(0o644)
    state = folder / "state"
    state.mkdir(mode=0o700)
    account = ubuntu()
    os.chown(state, account.pw_uid, account.pw_gid)
    command = service_command(attempt, deadline, folder / "detached.py")
    subprocess.run(command, check=True, timeout=min(20, remaining(deadline)))
    print(json.dumps({"attempt": attempt, "unit": specification["unit"], "installation": "started"}), flush=True)


def request(attempt):
    validate(attempt, 1)
    folder = LOCAL_ROOT / attempt
    spec = json.loads((folder / "request.json").read_text())
    validate(spec["attempt"], spec["deadline_unix"])
    if spec["attempt"] != attempt or spec["unit"] != unit_name(attempt):
        raise ValueError("Worker request identity differs")
    if digest((folder / "payload.sh").read_bytes()) != spec["payload_sha256"]:
        raise ValueError("Worker payload changed after installation")
    if digest(Path(__file__).read_bytes()) != spec["helper_sha256"]:
        raise ValueError("Worker helper changed after installation")
    return spec, folder


def mounted():
    result = subprocess.run(
        ["findmnt", "--noheadings", "--raw", "--mountpoint", str(PERSISTENT_ROOT), "--output", "TARGET"],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    return result.returncode == 0 and result.stdout.strip() == str(PERSISTENT_ROOT)


def persistent_directory(attempt):
    validate(attempt, 1)
    # Never create the expected NFS path on the ephemeral root filesystem.
    if not mounted():
        return None
    parent = PERSISTENT_ROOT / "data/operations/detached"
    for path in (PERSISTENT_ROOT, PERSISTENT_ROOT / "data", parent.parent, parent):
        if path.is_symlink():
            raise ValueError("Persistent evidence directories cannot be symlinks")
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / attempt
    if folder.is_symlink():
        raise ValueError("Persistent attempt directory cannot be a symlink")
    folder.mkdir(mode=0o700, exist_ok=True)
    return folder


def record(spec, local, persistent, status, **details):
    value = {**spec, "observed_unix": time.time(), "status": status, "worker_uid": os.geteuid(), **details}
    for folder in (local, persistent):
        if folder is not None:
            write_json(folder / "status.json", value)
            with (folder / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
    return value


class WorkerStopped(BaseException):
    def __init__(self, number):
        self.number = number


def stop_child(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
    except ProcessLookupError:
        pass


def run_command(command, output, deadline):
    environment = {key: value for key, value in os.environ.items() if key not in SECRET_ENV}
    remaining(deadline)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    try:
        while process.poll() is None:
            try:
                process.wait(timeout=min(2, remaining(deadline)))
            except subprocess.TimeoutExpired:
                continue
        return process.returncode if process.returncode >= 0 else 128 - process.returncode
    finally:
        stop_child(process)


def worker(attempt):
    spec, folder = request(attempt)
    if os.geteuid() != ubuntu().pw_uid:
        raise PermissionError("The worker must run as ubuntu")
    local, persistent = folder / "state", None
    if (local / "started.json").exists():
        raise ValueError("A worker attempt may execute only once")
    with (local / "started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)

    def interrupted(number, _frame):
        raise WorkerStopped(number)

    handlers = {number: signal.signal(number, interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    code, reason = 1, "bootstrap_failed"
    try:
        record(spec, local, None, "waiting_for_cloud_init")
        with (local / "bootstrap.log").open("ab", buffering=0) as output:
            code = run_command(["sudo", "-n", "cloud-init", "status", "--wait"], output, spec["deadline_unix"])
        if code:
            return code
        record(spec, local, None, "waiting_for_persistent_mount")
        while persistent is None:
            remaining(spec["deadline_unix"])
            persistent = persistent_directory(attempt)
            if persistent is None:
                time.sleep(min(2, remaining(spec["deadline_unix"])))
        if (persistent / "request.json").exists() or (persistent / "result.json").exists():
            # Do not let the failure receipt overwrite a previous attempt's files.
            persistent = None
            raise ValueError("Persistent attempt evidence already exists; refusing another worker")
        write_json(persistent / "request.json", spec)
        (persistent / "bootstrap.log").write_bytes((local / "bootstrap.log").read_bytes())
        record(spec, local, persistent, "running", persistent_directory=str(persistent))
        with (persistent / "worker.log").open("xb", buffering=0) as output:
            code = run_command(["/bin/bash", str(folder / "payload.sh")], output, spec["deadline_unix"])
        reason = "completed" if code == 0 else "payload_failed"
        return code
    except TimeoutError:
        code, reason = 124, "deadline_reached"
        return code
    except WorkerStopped as exc:
        code = 128 + exc.number
        reason = "service_stopped"
        return code
    except Exception as exc:
        code, reason = 1, "worker_failed"
        record(spec, local, persistent, reason, error_type=type(exc).__name__, error=str(exc))
        return code
    finally:
        value = record(spec, local, persistent, reason, exit_code=code)
        for location in (local, persistent):
            if location is not None:
                write_json(location / "result.json", value)
        for number, handler in handlers.items():
            signal.signal(number, handler)


def finalize(attempt):
    """ExecStopPost preserves systemd's result even after a forced worker exit."""
    spec, folder = request(attempt)
    persistent = persistent_directory(attempt)
    if (
        persistent is not None
        and (persistent / "request.json").exists()
        and json.loads((persistent / "request.json").read_text()) != spec
    ):
        raise ValueError("Persistent receipt belongs to a different worker specification")
    local = folder / "state"
    service = {key: os.environ.get(key) for key in ("SERVICE_RESULT", "EXIT_CODE", "EXIT_STATUS")}
    raw_code = service["EXIT_STATUS"]
    code = int(raw_code) if service["EXIT_CODE"] == "exited" and (raw_code or "").isdigit() else 1
    if (local / "result.json").exists():
        value = json.loads((local / "result.json").read_text())
    else:
        value = record(spec, local, persistent, "service_ended_without_worker_receipt", exit_code=code)
    for location in (local, persistent):
        if location is not None:
            write_json(location / "service.json", {**value, "systemd": service, "recorded_unix": time.time()})
            if not (location / "result.json").exists():
                write_json(location / "result.json", value)


def monitor(attempt):
    spec, folder = request(attempt)
    local = folder / "state"
    positions = {}
    while time.time() <= spec["deadline_unix"] + 20:
        paths = [local / "bootstrap.log", PERSISTENT_ROOT / f"data/operations/detached/{attempt}/worker.log"]
        for path in paths:
            if path.is_file():
                with path.open("rb") as stream:
                    stream.seek(positions.get(path, 0))
                    data = stream.read(1024 * 1024)
                    positions[path] = stream.tell()
                if data:
                    sys.stdout.buffer.write(data)
                    sys.stdout.buffer.flush()
        result = local / "result.json"
        if result.is_file():
            return json.loads(result.read_text())["exit_code"]
        time.sleep(1)
    raise TimeoutError("No worker exit receipt before the fixed monitoring deadline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "worker", "monitor", "finalize"))
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--script", type=Path)
    args = parser.parse_args()
    if args.action == "install":
        if args.deadline_unix is None or args.script is None:
            parser.error("install requires --deadline-unix and --script")
        install(args.attempt, args.deadline_unix, args.script)
        return 0
    if args.action == "worker":
        return worker(args.attempt)
    if args.action == "monitor":
        return monitor(args.attempt)
    finalize(args.attempt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
