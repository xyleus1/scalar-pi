"""User-operated Lambda sessions. Dry runs neither authenticate nor create resources.

Only this Windows-side controller needs the API credential. Instances receive a
public SSH key and an ordinary shell script, never the Lambda key. Keep this
process and the computer awake until termination is confirmed.
"""

import argparse
import base64
from contextlib import contextmanager
import datetime as dt
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
import uuid

API = "https://cloud.lambda.com/api/v1/"
REPOSITORY = "https://github.com/xyleus1/scalar-pi.git"
REMOTE_ROOT = "/lambda/nfs/scalar-pi"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LambdaApiError(RuntimeError):
    def __init__(self, method, endpoint, status):
        super().__init__(f"Lambda {method} {endpoint} returned HTTP {status}")
        self.status = status


class LambdaClient:
    def __init__(self, key):
        if not key:
            raise ValueError("Load LAMBDA_API_KEY using the private PowerShell loader")
        self.key = key

    def request(self, method, endpoint, body=None):
        if not re.fullmatch(r"[a-z0-9/-]+", endpoint):
            raise ValueError("Invalid Lambda API endpoint")
        request = urllib.request.Request(
            API + endpoint,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=30) as response:
                return json.load(response)["data"]
        except urllib.error.HTTPError as exc:
            # Never print request headers or an untrusted response body.
            raise LambdaApiError(method, endpoint, exc.code) from None
        except urllib.error.URLError:
            raise RuntimeError(f"Lambda {method} {endpoint} could not connect") from None


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_state(path):
    if not path.exists():
        return {"schema_version": 1, "resources": {}, "runs": []}
    state = json.loads(path.read_text(encoding="utf-8-sig"))
    if state.get("schema_version") != 1 or not isinstance(state.get("runs"), list):
        raise ValueError("Invalid project ledger; do not replace or reset it")
    return state


@contextmanager
def project_lock(path):
    """OS-released lock: a crashed controller cannot leave a stale lock held."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError("Another project controller is active; do not launch a second session") from None
        yield


def finite_positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return value


def validate_config(config):
    if config["instance_type"] != "gpu_1x_h100_pcie" or config["region"] != "us-west-3":
        raise ValueError("This protocol pins H100 PCIe in us-west-3; revise the protocol before changing hardware")
    for key in ("budget_usd", "qualification_budget_usd", "max_hourly_rate_usd"):
        finite_positive(config[key], key)
    if config["budget_usd"] > 2000 or config["qualification_budget_usd"] > 100:
        raise ValueError("This study is authorized for $2,000, with $100 for qualification")
    for key in ("noncompute_spent_usd", "noncompute_reserve_usd"):
        finite_positive(config[key], key, zero=True)


def session_cost(run, now=None):
    end = run.get("ended_unix") or (time.time() if now is None else now)
    # Count from launch request and round up: conservative relative to billable activation.
    minutes = math.ceil(max(0, end - run["started_unix"]) / 60)
    return minutes / 60 * run["hourly_rate_usd"]


def check_allowance(config, state, hours, rate, stage, budget_report=None):
    finite_positive(hours, "max_hours")
    finite_positive(rate, "hourly_rate")
    spent = sum(session_cost(run) for run in state["runs"]) + config["noncompute_spent_usd"]
    forecast = spent + config["noncompute_reserve_usd"] + 1.2 * hours * rate
    if forecast > config["budget_usd"]:
        raise ValueError("Session lifetime plus reserve exceeds the remaining project budget")
    if stage == "qualification":
        phase_spent = sum(session_cost(run) for run in state["runs"] if run["stage"] == stage)
        if phase_spent + config["noncompute_spent_usd"] + 1.2 * hours * rate > config["qualification_budget_usd"]:
            raise ValueError("Qualification session exceeds the $100 phase allowance")
    else:
        if not budget_report or budget_report.get("selected_study") not in {"full", "pilot"}:
            raise ValueError("Research requires a measured full/pilot budget decision")
        if budget_report.get("budget_usd") != config["budget_usd"] or budget_report.get("hourly_rate_usd") != rate:
            raise ValueError("Budget report ceiling/rate differs from this session; regenerate it")
        remaining = budget_report.get("remaining_compute_cost_with_headroom_usd")
        finite_positive(remaining, "remaining_compute_cost_with_headroom_usd", zero=True)
        if spent + config["noncompute_reserve_usd"] + remaining > config["budget_usd"]:
            raise ValueError("Measured study no longer fits the remaining balance; regenerate its budget decision")
    return {"spent_usd": spent, "session_ceiling_usd": hours * rate, "with_reserve_usd": forecast}


def select_type(client, config):
    item = client.request("GET", "instance-types")[config["instance_type"]]
    instance_type = item["instance_type"]
    rate = instance_type["price_cents_per_hour"] / 100
    if instance_type["specs"]["gpus"] != 1 or instance_type["architecture"] != "x86_64":
        raise ValueError("Expected one x86-64 H100 instance")
    if rate > config["max_hourly_rate_usd"]:
        raise ValueError("Current quote exceeds configured maximum; inspect and revise the cost estimate")
    if config["region"] not in {region["name"] for region in item["regions_with_capacity_available"]}:
        raise ValueError("Selected region has no capacity; no fallback instance will be launched")
    return rate


def prepare(client, config, state, state_path):
    if config.get("credits_verified") is not True:
        raise ValueError("Verify credit eligibility/expiry and storage pricing before setting credits_verified")
    select_type(client, config)
    spent = sum(session_cost(run) for run in state["runs"]) + config["noncompute_spent_usd"]
    if spent + config["noncompute_reserve_usd"] >= config["budget_usd"]:
        raise ValueError("No remaining budget for persistent storage")
    cidr = ipaddress.ip_network(config["operator_cidr"], strict=True)
    if cidr.version != 4 or cidr.prefixlen != 32:
        raise ValueError("operator_cidr must be the operator's IPv4 /32")
    key = Path(config["ssh_key_path"]).expanduser()
    key.parent.mkdir(parents=True, exist_ok=True)
    if not key.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "scalar-pi", "-f", str(key)], check=True)
    public_key = Path(str(key) + ".pub").read_text(encoding="utf-8").strip()
    if not public_key.startswith("ssh-ed25519 "):
        raise ValueError("Expected the project Ed25519 public key")
    resources = state["resources"]
    if "ssh_key_id" not in resources:
        existing = client.request("GET", "ssh-keys")
        if any(item["name"] == "scalar-pi" for item in existing):
            raise ValueError("An unrecorded scalar-pi SSH key exists; reconcile it instead of replacing it")
        resources["ssh_key_id"] = client.request("POST", "ssh-keys", {"name": "scalar-pi", "public_key": public_key})[
            "id"
        ]
        save_state(state_path, state)
    if "filesystem_id" not in resources:
        existing = client.request("GET", "filesystems")
        if any(item["name"] == "scalar-pi" for item in existing):
            raise ValueError("An unrecorded scalar-pi filesystem exists; reconcile it instead of adopting it")
        resources["filesystem_id"] = client.request(
            "POST", "filesystems", {"name": "scalar-pi", "region": config["region"]}
        )["id"]
        save_state(state_path, state)
    print("Project SSH key and persistent filesystem recorded. No GPU launched.")


def launch_body(config, state, commit, name):
    if re.fullmatch(r"[a-f0-9]{40}", commit or "") is None:
        raise ValueError("An immutable 40-character repository commit is required")
    if not re.fullmatch(r"[a-f0-9-]{32,36}", config["image_id"]):
        raise ValueError("Select a real x86-64 GPU Base 22.04 image ID from inspect")
    cidr = ipaddress.ip_network(config["operator_cidr"], strict=True)
    if cidr.version != 4 or cidr.prefixlen != 32:
        raise ValueError("operator_cidr must be one IPv4 /32")
    # Host rules enforce restriction even if the shared account firewall is broader.
    user_data = (
        "#cloud-config\npackages: [ufw, git]\nruncmd:\n"
        "  - [ufw, default, deny, incoming]\n"
        f"  - [ufw, allow, from, '{cidr}', to, any, port, '22', proto, tcp]\n"
        "  - [ufw, --force, enable]\n"
    )
    return {
        "region_name": config["region"],
        "instance_type_name": config["instance_type"],
        "ssh_key_names": ["scalar-pi"],
        "file_system_names": ["scalar-pi"],
        "image": {"id": config["image_id"]},
        "name": name,
        "user_data": user_data,
        "tags": [{"key": "project", "value": "scalar-pi"}, {"key": "commit", "value": commit}],
    }


def terminate(client, state, state_path, instance_id):
    if state.get("pending_launch"):
        pending = state["pending_launch"]
        candidate = client.request("GET", "instances/" + instance_id)
        if candidate.get("name") != pending["name"]:
            raise ValueError("Instance does not match the recorded uncertain launch")
        state["runs"].append({**pending, "instance_id": instance_id})
        del state["pending_launch"]
        save_state(state_path, state)
    matches = [run for run in state["runs"] if run["instance_id"] == instance_id]
    if len(matches) != 1:
        raise ValueError("Refusing to terminate an ID not uniquely recorded in this project's ledger")
    run = matches[0]
    if run.get("ended_unix"):
        print("Termination was already confirmed; recorded cost is unchanged.")
        return
    try:
        current = client.request("GET", "instances/" + instance_id)
    except LambdaApiError as exc:
        if exc.status != 404:
            raise
        current = {"name": run["name"], "status": "terminated"}
    if current.get("name") != run["name"]:
        raise ValueError("Recorded instance identity does not match Lambda")
    if current["status"] != "terminated":
        client.request("POST", "instance-operations/terminate", {"instance_ids": [instance_id]})
    for _ in range(60):
        try:
            current = client.request("GET", "instances/" + instance_id)
        except LambdaApiError as exc:
            if exc.status != 404:
                raise
            current = {"status": "terminated"}
        if current["status"] == "terminated":
            run["ended_unix"] = time.time()
            run["estimated_compute_usd"] = session_cost(run)
            save_state(state_path, state)
            print("Termination confirmed; persistent storage remains billable.")
            return
        time.sleep(5)
    raise RuntimeError("Termination not confirmed. Check Lambda immediately; ledger remains open.")


def remote_script(commit, script):
    root = shlex.quote(REMOTE_ROOT)
    return (
        "set -euo pipefail\nsudo cloud-init status --wait\n"
        f"test -d {root}\ncd {root}\n"
        f"if [ ! -d repo/.git ]; then git clone {shlex.quote(REPOSITORY)} repo; fi\n"
        'cd repo\ntest -z "$(git status --porcelain)"\n'
        f"git fetch origin {commit}\ngit checkout --detach {commit}\n"
        "git submodule update --init --recursive\n" + script.replace("\r\n", "\n") + "\n"
    )


def run_session(client, config, state, state_path, args):
    if config.get("credits_verified") is not True:
        raise ValueError("Credits must be verified before launch")
    if state.get("pending_launch") or any(not run.get("ended_unix") for run in state["runs"]):
        raise ValueError("A recorded session is still open; reconcile/terminate it first")
    if not {"ssh_key_id", "filesystem_id"}.issubset(state["resources"]):
        raise ValueError("Run prepare first")
    rate = select_type(client, config)
    report = json.loads(args.budget_report.read_text()) if args.budget_report else None
    if report and "budget" in report:
        report = report["budget"]
    allowance = check_allowance(config, state, args.max_hours, rate, args.stage, report)
    if report and report.get("experiment_identity", {}).get("code_commit", args.commit) != args.commit:
        raise ValueError("Budget was measured for a different source commit")
    images = client.request("GET", "images")
    if not any(
        image["id"] == config["image_id"]
        and image["architecture"] == "x86_64"
        and image["region"]["name"] == config["region"]
        and image["family"] == "gpu-base-22-04"
        for image in images
    ):
        raise ValueError("Configured image is not an available x86-64 GPU Base 22.04 image in this region")
    name = "scalar-pi-" + uuid.uuid4().hex[:12]
    body = launch_body(config, state, args.commit, name)
    started = time.time()
    exports = (
        f"export SCALAR_PI_HOURLY_RATE_USD={rate}\n"
        f"export SCALAR_PI_DEADLINE_UNIX={started + args.max_hours * 3600}\n"
        f"export SCALAR_PI_ATTEMPT={name}\n"
    )
    if args.budget_report:
        # Bind the worker's study decision to the same report checked on Windows.
        encoded = base64.b64encode(args.budget_report.read_bytes()).decode("ascii")
        exports += (
            f"mkdir -p {REMOTE_ROOT}/data/experiments/reports\n"
            f"printf '%s' '{encoded}' | base64 --decode > {REMOTE_ROOT}/data/experiments/reports/budget.json\n"
        )
    script = remote_script(args.commit, exports + args.script.read_text(encoding="utf-8-sig"))
    pending = {
        "name": name,
        "commit": args.commit,
        "started_unix": started,
        "hourly_rate_usd": rate,
        "stage": args.stage,
        "max_hours": args.max_hours,
        "allowance": allowance,
    }
    state["pending_launch"] = pending
    save_state(state_path, state)
    try:
        launched = client.request("POST", "instance-operations/launch", body)
    except BaseException as exc:
        if isinstance(exc, LambdaApiError) and 400 <= exc.status < 500:
            del state["pending_launch"]
            save_state(state_path, state)
            raise
        # A timed-out POST may have launched a GPU. Never retry a launch blindly.
        candidates = [item for item in client.request("GET", "instances") if item.get("name") == name]
        if len(candidates) == 1:
            terminate(client, state, state_path, candidates[0]["id"])
        raise RuntimeError(
            f"Launch response uncertain. Inspect {name} in Lambda; pending ledger blocks another launch"
        ) from None
    instance_id = launched["instance_ids"][0]
    run = {**pending, "instance_id": instance_id}
    state["runs"].append(run)
    del state["pending_launch"]
    deadline = started + args.max_hours * 3600
    process = None
    try:
        save_state(state_path, state)
        while time.time() < deadline:
            instance = client.request("GET", "instances/" + instance_id)
            if instance["status"] == "active" and instance.get("ip"):
                break
            if instance["status"] in {"unhealthy", "terminated", "preempted"}:
                raise RuntimeError("Instance did not become healthy")
            time.sleep(5)
        else:
            raise TimeoutError("Session deadline reached during boot")
        ipaddress.ip_address(instance["ip"])
        log = state_path.parent / (name + ".log")
        ssh_options = [
            "-i",
            config["ssh_key_path"],
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"UserKnownHostsFile={state_path.parent / 'known_hosts'}",
        ]
        # Transfer first, then execute with stdin closed. Interactive Docker
        # commands must not consume the remaining shell program from bash -s.
        receiver = 'script=$(mktemp); trap \'rm -f -- "$script"\' EXIT; cat > "$script"; bash "$script" </dev/null'
        command = ["ssh", *ssh_options, "ubuntu@" + instance["ip"], "bash -c " + shlex.quote(receiver)]
        print(f"Running {name}; output: {log}. Keep this controller awake until termination is confirmed.")
        environment = {key: value for key, value in os.environ.items() if key != "LAMBDA_API_KEY"}
        ssh_deadline = min(deadline, time.time() + 120)
        while time.time() < ssh_deadline:
            ready = subprocess.run(
                ["ssh", *ssh_options, "ubuntu@" + instance["ip"], "true"],
                env=environment,
                capture_output=True,
                timeout=max(1, min(35, ssh_deadline - time.time())),
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(min(5, max(0, ssh_deadline - time.time())))
        else:
            raise TimeoutError("SSH did not become ready within the session allowance")
        with log.open("wb") as output:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=subprocess.STDOUT,
                env=environment,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            process.stdin.write(script.encode())
            process.stdin.close()
            next_quote_check = time.time() + 60
            while process.poll() is None:
                if time.time() >= deadline:
                    raise TimeoutError("Session deadline reached; terminating GPU")
                if time.time() >= next_quote_check:
                    quote = client.request("GET", "instance-types")[config["instance_type"]]["instance_type"]
                    current_rate = quote["price_cents_per_hour"] / 100
                    run["hourly_rate_usd"] = max(run["hourly_rate_usd"], current_rate)
                    save_state(state_path, state)
                    if current_rate > rate:
                        raise RuntimeError("Provider price rose above the budgeted rate; terminating GPU")
                    next_quote_check = time.time() + 60
                time.sleep(1)
        if deadline - time.time() > 5:
            destination = state_path.parent / "reports" / name
            destination.mkdir(parents=True, exist_ok=True)
            transfer = [
                "scp",
                "-r",
                *ssh_options,
                "ubuntu@" + instance["ip"] + ":" + REMOTE_ROOT + "/data/experiments/reports/.",
                str(destination),
            ]
            try:
                result = subprocess.run(
                    transfer, env=environment, capture_output=True, timeout=min(60, deadline - time.time()), check=False
                )
                run["reports_downloaded"] = result.returncode == 0
            except (OSError, subprocess.TimeoutExpired):
                run["reports_downloaded"] = False
        if process.returncode:
            raise RuntimeError(f"Remote script failed with exit {process.returncode}; see {log}")
    finally:
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        finally:
            terminate(client, state, state_path, instance_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("action", choices=("inspect", "prepare", "run", "terminate"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--script", type=Path)
    parser.add_argument("--commit")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--stage", choices=("qualification", "research"), default="qualification")
    parser.add_argument("--budget-report", type=Path)
    parser.add_argument("--instance-id")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    validate_config(config)
    if args.action != "inspect" and not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "action": args.action,
                    "region": config["region"],
                    "instance_type": config["instance_type"],
                    "stage": args.stage,
                    "max_hours": args.max_hours,
                    "script": str(args.script),
                    "commit": args.commit,
                    "resources_created": 0,
                },
                indent=2,
            )
        )
        return 0
    client = LambdaClient(os.environ.get("LAMBDA_API_KEY"))
    state_path = args.config.parent / "lambda-state.json"
    if args.action == "inspect":
        state = load_state(state_path)
        types = client.request("GET", "instance-types")
        print(
            json.dumps(
                {
                    "observed_utc": dt.datetime.now(dt.UTC).isoformat(),
                    "selected_type": types.get(config["instance_type"]),
                    "images": client.request("GET", "images"),
                    "active_project_sessions": [run for run in state["runs"] if not run.get("ended_unix")],
                    "pending_launch": state.get("pending_launch"),
                    "credit_balance_verified_by_api": False,
                },
                indent=2,
            )
        )
    else:
        with project_lock(state_path.with_suffix(".lock")):
            state = load_state(state_path)
            if args.action == "prepare":
                prepare(client, config, state, state_path)
            elif args.action == "terminate":
                terminate(client, state, state_path, args.instance_id)
            else:
                if not args.script or not args.script.is_file() or args.max_hours is None:
                    parser.error("run requires --script and --max-hours")
                run_session(client, config, state, state_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
