"""Launch once under an already running remote deadline watchdog, then exit.

Pre-arm github_guard.py before minting the short-lived workspace-member Lambda key.
The master key performs read-only preflight; only the expiring key can launch.
No SSH connection, local monitor, or finally block owns instance termination.
"""

import argparse
import base64
import datetime as dt
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import time

import cloud
import detached
from watchdog import WORKER_COMMIT
from watchdog import sha256
from watchdog import validate_request

USER_DATA_LIMIT = 1_000_000
LAUNCH_ROLE_ID = "13be71019d86427e853e4dafe915384d"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_key(key, receipt, armed, *, now, master_key):
    if not isinstance(key, str) or not key or not master_key or hmac.compare_digest(key, master_key):
        raise ValueError("Launch requires a separate expiring workspace-member key")
    if receipt.get("schema_version") != 1:
        raise ValueError("Unsupported creator-key receipt")
    for field in ("identity_id", "key_id"):
        value = receipt.get(field)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value):
            raise ValueError("Creator-key receipt requires provider identity and key IDs")
    if receipt.get("workspace_id") != armed["expected"]["workspace_id"]:
        raise ValueError("Creator key belongs to a different workspace")
    if receipt.get("role_ids") != [LAUNCH_ROLE_ID] or receipt.get("role_names") != ["workspace-member"]:
        raise ValueError("Creator-key receipt requires the audited workspace-member role")
    if receipt.get("workspace_ids") != [receipt["workspace_id"]] or any(
        receipt.get(field) != [] for field in ("account_role_ids", "inherited_role_ids", "group_ids")
    ):
        raise ValueError("Creator key must have exactly one workspace and no account, inherited, or group grants")
    permissions = receipt.get("permission_ids")
    if (
        not isinstance(permissions, list)
        or any(not isinstance(value, str) or not value for value in permissions)
        or len(permissions) != len(set(permissions))
        or "compute:instance:launch" not in permissions
    ):
        raise ValueError("Creator-key receipt requires the full audited permissions including compute:instance:launch")
    digest = receipt.get("key_sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Creator-key receipt requires the raw key SHA256")
    if not hmac.compare_digest(hashlib.sha256(key.encode()).hexdigest(), digest):
        raise ValueError("Creator key differs from its audited receipt")
    try:
        expires = dt.datetime.fromisoformat(receipt["expires_at"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, TypeError, ValueError):
        raise ValueError("Creator-key receipt requires the actual provider expires_at") from None
    if expires.tzinfo is None:
        raise ValueError("Creator-key expiry must include its timezone")
    expiry = expires.timestamp()
    if not now + 60 < expiry <= armed["launch_not_after_unix"]:
        raise ValueError("Creator key must expire within the armed launch window and allow more than sixty seconds")
    return {
        field: receipt[field]
        for field in (
            "schema_version",
            "identity_id",
            "key_id",
            "workspace_id",
            "workspace_ids",
            "role_ids",
            "role_names",
            "account_role_ids",
            "inherited_role_ids",
            "group_ids",
            "permission_ids",
            "expires_at",
            "key_sha256",
        )
    }


def bind_guard(armed, request):
    request, deadline = validate_request(request, armed["armed_unix"])
    expected = {
        key: request[key] for key in ("instance_name", "workspace_id", "worker_commit", "region", "instance_type")
    }
    github = armed["github"]
    if not all(str(github[key]).isdigit() for key in ("run_id", "run_attempt")):
        raise ValueError("Invalid watchdog run identity")
    expected["tags"] = {
        "project": "scalar-pi",
        "commit": WORKER_COMMIT,
        "watchdog-run": str(github["run_id"]) + "-" + str(github["run_attempt"]),
        "watchdog-deadline": str(deadline),
    }
    if (
        armed.get("dry_run") is not False
        or armed.get("status") != "armed"
        or armed.get("request") != request
        or armed.get("request_sha256") != sha256(request)
        or armed.get("expected") != expected
        or armed.get("deadline_unix") != deadline
        or armed.get("launch_not_after_unix") != armed["armed_unix"] + request["launch_window_seconds"]
    ):
        raise ValueError("Live watchdog does not match the exact launch request")
    return deadline


def build_body(config, state, armed, script, report_bytes, *, rate, forecast_rate):
    deadline = armed["deadline_unix"]
    name = armed["expected"]["instance_name"]
    exports = (
        f"export SCALAR_PI_HOURLY_RATE_USD={rate}\n"
        f"export SCALAR_PI_FORECAST_RATE_USD={forecast_rate}\n"
        f"export SCALAR_PI_DEADLINE_UNIX={deadline}\n"
        f"export SCALAR_PI_ATTEMPT={name}\n"
    )
    if report_bytes is not None:
        encoded = base64.b64encode(report_bytes).decode("ascii")
        exports += (
            f"mkdir -p {cloud.REMOTE_ROOT}/data/experiments/reports\n"
            f"printf '%s' '{encoded}' | base64 --decode > {cloud.REMOTE_ROOT}/data/experiments/reports/budget.json\n"
        )
    worker = cloud.remote_script(WORKER_COMMIT, exports + script, with_submodules=True)
    bootstrap = detached.build_remote_script(worker, attempt=name, deadline_unix=deadline, bootstrap_only=True)
    body = cloud.launch_body(config, state, WORKER_COMMIT, name)
    body["workspace_id"] = armed["expected"]["workspace_id"]
    body["tags"] = [{"key": key, "value": value} for key, value in armed["expected"]["tags"].items()]
    body["user_data"] += "  - " + json.dumps(["bash", "-c", bootstrap]) + "\n"
    if len(body["user_data"].encode("utf-8")) > USER_DATA_LIMIT:
        raise ValueError("Complete UTF-8 cloud-init user_data exceeds the one-million-byte limit")
    return body


def launch(
    config,
    state_path,
    script_path,
    key_receipt_path,
    guard_path,
    request,
    guard,
    *,
    reader,
    launch_key,
    stage="qualification",
    budget_report_path=None,
    now=time.time,
):
    """Use prepared resources and one launch POST. All ambiguous outcomes stay pending."""
    cloud.validate_config(config)
    if stage not in {"qualification", "research"}:
        raise ValueError("Guarded launch supports qualification and research only")
    if (config["instance_type"], config["region"]) != cloud.RESEARCH_PROFILE:
        raise ValueError("Guarded launch requires the original research hardware profile")
    cloud.require_billing(config, stage)
    script_bytes = script_path.read_bytes()
    script = script_bytes.decode("utf-8-sig")
    receipt_bytes = key_receipt_path.read_bytes()
    key_receipt = json.loads(receipt_bytes.decode("utf-8-sig"))
    report_bytes = budget_report_path.read_bytes() if budget_report_path else None
    report = json.loads(report_bytes.decode("utf-8-sig")) if report_bytes is not None else None
    if report and "budget" in report:
        report = report["budget"]
    if report and report.get("experiment_identity", {}).get("code_commit", WORKER_COMMIT) != WORKER_COMMIT:
        raise ValueError("Budget was measured for a different scientific source commit")
    master_key = reader.key
    if any(
        secret and (secret in script or (report_bytes is not None and secret.encode() in report_bytes))
        for secret in (master_key, launch_key)
    ):
        raise ValueError("Worker inputs must not contain cloud credentials")
    with cloud.project_lock(state_path.with_suffix(".lock")):
        state = cloud.load_state(state_path)
        if state.get("pending_launch") or any(not run.get("ended_unix") for run in state["runs"]):
            raise ValueError("A recorded session is still open; reconcile it before launching")
        if any(run.get("name") == request["instance_name"] for run in state["runs"]):
            raise ValueError("An attempt name already in the ledger cannot be reused")
        if not {"ssh_key_id", "filesystem_id"}.issubset(state["resources"]):
            raise ValueError("Prepared project resources are required")
        armed = guard.wait_ready(guard_path, timeout_seconds=60)
        deadline = bind_guard(armed, request)
        validate_key(launch_key, key_receipt, armed, now=now(), master_key=master_key)
        rate = cloud.select_type(reader, config)
        images = reader.request("GET", "images")
        if not any(
            image["id"] == config["image_id"]
            and image["architecture"] == "x86_64"
            and image["region"]["name"] == config["region"]
            and image["family"] == "gpu-base-22-04"
            for image in images
        ):
            raise ValueError("Configured image is not an available x86-64 GPU Base 22.04 image in this region")
        instances = reader.request("GET", "instances")
        if not isinstance(instances, list):
            raise ValueError("Unexpected instance-list response")
        if any(
            item.get("name") == request["instance_name"]
            or (
                item.get("workspace_id") == request["workspace_id"]
                and str(item.get("name", "")).startswith("scalar-pi-")
                and item.get("status") != "terminated"
            )
            for item in instances
        ):
            raise ValueError("An existing project instance requires reconciliation before launch")
        if stage == "research":
            if not report:
                raise ValueError("Research requires its measured budget and study decision")
            cloud.bind_study(state, report, WORKER_COMMIT)
        checked_at = now()
        hours = (deadline - checked_at) / 3600
        cloud.session_deadline(config, stage, hours, checked_at)
        allowance = cloud.check_allowance(config, state, hours, rate, stage, report)
        body = build_body(
            config, state, armed, script, report_bytes, rate=rate, forecast_rate=allowance["forecast_hourly_rate_usd"]
        )
        creator = cloud.LambdaClient(launch_key)
        # All read-only cloud preflight and payload construction precede this
        # live check. It never dispatches another guard or extends the deadline.
        live = guard.revalidate(guard_path, expected_request=request)
        if live != armed:
            raise ValueError("Armed receipt changed during launch preparation")
        started = now()
        hours = (deadline - started) / 3600
        cloud.session_deadline(config, stage, hours, started)
        audited_key = validate_key(launch_key, key_receipt, live, now=started, master_key=master_key)
        allowance = cloud.check_allowance(config, state, hours, rate, stage, report)
        pending = {
            "name": request["instance_name"],
            "commit": WORKER_COMMIT,
            "instance_type": config["instance_type"],
            "region": config["region"],
            "workspace_id": request["workspace_id"],
            "filesystem_id": state["resources"]["filesystem_id"],
            "filesystem_mount_point": cloud.REMOTE_ROOT,
            "started_unix": started,
            "deadline_unix": deadline,
            "hourly_rate_usd": rate,
            "stage": stage,
            "max_hours": hours,
            "allowance": allowance,
            "watchdog": live,
            "creator_key": audited_key,
            "creator_key_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "script_sha256": hashlib.sha256(script_bytes).hexdigest(),
            "budget_report_sha256": hashlib.sha256(report_bytes).hexdigest() if report_bytes else None,
            "launch_body_sha256": sha256(body),
        }
        state["pending_launch"] = pending
        cloud.save_state(state_path, state)
        try:
            # Even suspension after the live check cannot extend admission:
            # the provider rejects this creator key after its actual expiry.
            validate_key(launch_key, key_receipt, live, now=now(), master_key=master_key)
        except BaseException:
            del state["pending_launch"]
            cloud.save_state(state_path, state)
            raise
        try:
            launched = creator.request("POST", "instance-operations/launch", body)
            ids = launched.get("instance_ids")
            if (
                not isinstance(ids, list)
                or len(ids) != 1
                or not isinstance(ids[0], str)
                or not re.fullmatch(r"(?:[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})", ids[0])
            ):
                raise ValueError("Launch response did not identify exactly one instance")
        except BaseException as error:
            if isinstance(error, cloud.LambdaApiError) and 400 <= error.status < 500 and error.status not in {408, 499}:
                del state["pending_launch"]
                cloud.save_state(state_path, state)
                raise
            raise RuntimeError(
                "Launch response uncertain; the pending ledger blocks retries and the remote watchdog owns cleanup"
            ) from None
        run = {**pending, "instance_id": ids[0]}
        state["runs"].append(run)
        del state["pending_launch"]
        cloud.save_state(state_path, state)
        return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True, help="The existing github_guard dispatch record")
    parser.add_argument("--key-receipt", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--budget-report", type=Path)
    parser.add_argument("--stage", choices=("qualification", "research"), default="qualification")
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--gh", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    request = read_json(args.request)
    cloud.validate_config(config)
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "stage": args.stage,
                    "worker_commit": WORKER_COMMIT,
                    "request": request,
                    "resources_created": 0,
                }
            )
        )
        return 0
    from github_guard import GitHubGuard

    guard = GitHubGuard(args.gh, args.evidence_root, args.controller_commit)
    run = launch(
        config,
        args.config.parent / "lambda-state.json",
        args.script,
        args.key_receipt,
        args.guard,
        request,
        guard,
        reader=cloud.LambdaClient(os.environ.get("LAMBDA_API_KEY")),
        launch_key=os.environ.get("LAMBDA_LAUNCH_API_KEY"),
        stage=args.stage,
        budget_report_path=args.budget_report,
    )
    print(
        json.dumps(
            {
                "instance_id": run["instance_id"],
                "name": run["name"],
                "deadline_unix": run["deadline_unix"],
                "watchdog": run["watchdog"]["github"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
