"""Launch once under an already running remote deadline watchdog, then exit.

Pre-arm github_guard.py before minting the short-lived workspace-member Lambda key.
The master key performs read-only preflight; only the expiring key can launch.
No SSH connection, local monitor, or finally block owns instance termination.
"""

import argparse
import base64
import copy
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
from watchdog import validate_instance
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


def build_body(
    config,
    state,
    armed,
    script,
    report_bytes,
    *,
    rate,
    forecast_rate,
    existing_checkout=False,
    normalization_sha256=None,
):
    deadline = armed["deadline_unix"]
    name = armed["expected"]["instance_name"]
    local_replica = cloud.storage_mode(config) == "replicated_local"
    if local_replica and (existing_checkout or normalization_sha256 is not None or report_bytes is not None):
        raise ValueError("Replicated local bootstrap is fresh host preparation only")
    exports = (
        f"export SCALAR_PI_HOURLY_RATE_USD={rate}\n"
        f"export SCALAR_PI_FORECAST_RATE_USD={forecast_rate}\n"
        f"export SCALAR_PI_DEADLINE_UNIX={deadline}\n"
        f"export SCALAR_PI_ATTEMPT={name}\n"
    )
    if local_replica:
        count = cloud.REPLICATED_PROFILES[(config["instance_type"], config["region"])]
        exports += f"export SCALAR_PI_EXPECTED_GPU_COUNT={count}\n"
        if config["instance_type"] in {"gpu_1x_h100_sxm5", "gpu_8x_h100_sxm5"}:
            exports += f"export SCALAR_PI_EXPECTED_INSTANCE_TYPE={config['instance_type']}\n"
    if report_bytes is not None:
        if existing_checkout:
            exports += (
                "python3 - <<'SCALAR_PI_VERIFY_BUDGET'\nimport hashlib\nfrom pathlib import Path\n"
                f"path = Path({(cloud.REMOTE_ROOT + '/data/experiments/reports/budget.json')!r})\n"
                "if path.is_symlink() or path.resolve() != path or not path.is_file():\n"
                "    raise ValueError('Existing study budget must be a regular nonsymlink file')\n"
                f"if hashlib.sha256(path.read_bytes()).hexdigest() != {hashlib.sha256(report_bytes).hexdigest()!r}:\n"
                "    raise ValueError('Existing study budget differs from launch admission')\n"
                "SCALAR_PI_VERIFY_BUDGET\n"
            )
        else:
            encoded = base64.b64encode(report_bytes).decode("ascii")
            exports += (
                f"mkdir -p {cloud.REMOTE_ROOT}/data/experiments/reports\n"
                f"printf '%s' '{encoded}' | base64 --decode > {cloud.REMOTE_ROOT}/data/experiments/reports/budget.json\n"
            )
    worker = cloud.remote_script(
        WORKER_COMMIT,
        exports + script,
        with_submodules=True,
        existing_checkout=existing_checkout,
        normalization_sha256=normalization_sha256,
    )
    bootstrap = detached.build_remote_script(worker, attempt=name, deadline_unix=deadline, bootstrap_only=True)
    body = cloud.launch_body(config, state, WORKER_COMMIT, name)
    body["workspace_id"] = armed["expected"]["workspace_id"]
    body["tags"] = [{"key": key, "value": value} for key, value in armed["expected"]["tags"].items()]
    if local_replica:
        from replicated_bootstrap import cloud_init_command

        body["user_data"] += "  - " + json.dumps(cloud_init_command(name, deadline)) + "\n"
    body["user_data"] += "  - " + json.dumps(["bash", "-c", bootstrap]) + "\n"
    if len(body["user_data"].encode("utf-8")) > USER_DATA_LIMIT:
        raise ValueError("Complete UTF-8 cloud-init user_data exceeds the one-million-byte limit")
    return body


def active_runs(state, protected_ids):
    """Opt-in names every existing VM; pending/unknown work still blocks launch."""
    if (
        not isinstance(protected_ids, tuple | list)
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[a-f0-9]{32}|[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", value) is None
            for value in protected_ids
        )
        or len(protected_ids) != len(set(protected_ids))
    ):
        raise ValueError("Protected active instance IDs must be an explicit unique list")
    opened = [run for run in state["runs"] if not run.get("ended_unix")]
    if state.get("pending_launch") or (opened and not protected_ids):
        raise ValueError("A recorded session is still open; reconcile it before launching")
    if len(opened) != len(protected_ids) or {run.get("instance_id") for run in opened} != set(protected_ids):
        raise ValueError("Protected IDs must match every open ledger run exactly")
    return opened


def verify_instances(instances, opened, request, *, allow_mixed_profiles=False):
    if not isinstance(instances, list):
        raise ValueError("Unexpected instance-list response")
    if any(item.get("name") == request["instance_name"] for item in instances):
        raise ValueError("An existing project instance requires reconciliation before launch")
    live = [
        item
        for item in instances
        if item.get("workspace_id") == request["workspace_id"]
        and str(item.get("name", "")).startswith("scalar-pi-")
        and item.get("status") != "terminated"
    ]
    if len(live) != len(opened) or {item.get("id") for item in live} != {run["instance_id"] for run in opened}:
        raise ValueError("An existing project instance requires reconciliation before launch")
    for run in opened:
        expected = run["watchdog"]["expected"]
        for field, other in (
            ("name", "instance_name"),
            ("commit", "worker_commit"),
            ("workspace_id", "workspace_id"),
            ("region", "region"),
            ("instance_type", "instance_type"),
        ):
            if run.get(field) != expected[other]:
                raise ValueError("Protected ledger identity differs from its watchdog binding")
        if allow_mixed_profiles:
            # Cross-region replicas share the workspace, not a filesystem or GPU
            # profile. Each existing VM remains bound to its own original guard.
            if (
                request.get("storage_mode") != "replicated_local"
                or (request["instance_type"], request["region"]) not in cloud.REPLICATED_PROFILES
                or expected["workspace_id"] != request["workspace_id"]
                or expected["worker_commit"] != WORKER_COMMIT
                or (expected["instance_type"], expected["region"])
                not in {cloud.RESEARCH_PROFILE, *cloud.REPLICATED_PROFILES}
            ):
                raise ValueError("Protected replica identity is outside the explicit workspace/source/profiles")
        elif any(expected[key] != request[key] for key in ("workspace_id", "worker_commit", "region", "instance_type")):
            raise ValueError("Parallel workers must share the same workspace/source/hardware profile")
        instance = next(item for item in live if item["id"] == run["instance_id"])
        if instance.get("status") != "active":
            raise ValueError("Protected instance must be active before admitting another worker")
        validate_instance(instance, expected, run["instance_id"])


def protected_coverage(opened, verifier, *, now, rate):
    """Validate a private verifier's fresh AWS proof; this module needs no AWS SDK.

    The callback receives copies of the exact locked ledger runs. It must GET
    each immutable AWS version, enabled schedule, key-expiry metadata and strong
    DynamoDB smoke/cleanup state, and prove the completed GitHub handover.
    Cached files alone are insufficient. Returned run_sha256 binds the result
    to the entire ledger entry; only this strict nonsecret receipt is retained.
    """
    if not opened:
        return None, {"open_compute_reserve_usd": 0, "open_qualification_reserve_usd": 0}
    if not callable(verifier):
        raise ValueError("Parallel admission requires a fresh private cleanup verifier")
    receipt = verifier(copy.deepcopy(opened))
    if (
        not isinstance(receipt, dict)
        or set(receipt) != {"schema_version", "observed_unix", "runs"}
        or receipt["schema_version"] != 1
    ):
        raise ValueError("Unsupported protected cleanup verification receipt")
    observed = cloud.finite_positive(receipt["observed_unix"], "Cleanup verification time")
    checked = now()
    if not 0 <= checked - observed <= 30:
        raise ValueError("Protected cleanup verification must be fresh within thirty seconds")
    proofs = receipt["runs"]
    if not isinstance(proofs, list) or len(proofs) != len(opened):
        raise ValueError("Cleanup verification must cover every open run")
    fields = {
        "instance_id",
        "run_sha256",
        "lease_id",
        "lease_sha256",
        "function_version_arn",
        "schedule_arn",
        "deadline_unix",
        "cleanup_deadline_unix",
        "key_expires_unix",
    }
    if any(not isinstance(proof, dict) or set(proof) != fields for proof in proofs):
        raise ValueError("Unexpected protected cleanup proof fields")
    if {proof["instance_id"] for proof in proofs} != {run["instance_id"] for run in opened}:
        raise ValueError("Cleanup verification instance IDs differ")
    total, qualification = 0.0, 0.0
    for run in opened:
        proof = next(item for item in proofs if item["instance_id"] == run["instance_id"])
        leases = [
            entry
            for entry in run.get("cleanup_leases", [])
            if entry.get("lease_id") == run.get("effective_cleanup_lease_id")
        ]
        if len(leases) != 1 or proof["run_sha256"] != sha256(run):
            raise ValueError("Cleanup proof differs from the effective ledger lease")
        lease = leases[0]
        if lease.get("kind") != "aws-lambda-scheduler" or lease.get("github_conclusion") != "cancelled":
            raise ValueError("Parallel admission requires a completed AWS cleanup handover")
        if any(proof[field] != lease.get(field) for field in fields - {"run_sha256"}):
            raise ValueError("Protected cleanup proof differs from the recorded AWS binding")
        deadline = cloud.finite_positive(proof["deadline_unix"], "Protected deadline")
        cleanup = cloud.finite_positive(proof["cleanup_deadline_unix"], "Protected cleanup deadline")
        expiry = cloud.finite_positive(proof["key_expires_unix"], "Protected key expiry")
        if not checked < deadline < cleanup <= expiry or cleanup - deadline != 7200:
            raise ValueError("Protected cleanup coverage is expired or incomplete")
        old_rate = cloud.finite_positive(run["hourly_rate_usd"], "Protected hourly rate")
        reserved = 1.2 * (cleanup - checked) / 3600 * max(old_rate, rate)
        total += reserved
        if run["stage"] in {"qualification", "canary"}:
            qualification += reserved
    return copy.deepcopy(receipt), {"open_compute_reserve_usd": total, "open_qualification_reserve_usd": qualification}


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
    protected_active_instance_ids=(),
    verify_active_leases=None,
    existing_checkout=False,
    normalization_sha256=None,
    now=time.time,
):
    """Use prepared resources and one launch POST. All ambiguous outcomes stay pending."""
    cloud.validate_config(config)
    local_replica = cloud.storage_mode(config) == "replicated_local"
    if any(config[field] != request[field] for field in ("region", "instance_type")):
        raise ValueError("Launch configuration differs from the pre-armed hardware/region")
    if request.get("storage_mode", "persistent_shared") != cloud.storage_mode(config):
        raise ValueError("Launch storage mode differs from the pre-armed request")
    if stage not in {"qualification", "research"}:
        raise ValueError("Guarded launch supports qualification and research only")
    if not local_replica and (config["instance_type"], config["region"]) != cloud.RESEARCH_PROFILE:
        raise ValueError("Guarded launch requires the original research hardware profile")
    if local_replica and (stage != "qualification" or existing_checkout or normalization_sha256 is not None):
        raise ValueError("Replicated local launch permits only a fresh qualification bootstrap")
    cloud.require_billing(config, stage)
    if protected_active_instance_ids and not existing_checkout and not local_replica:
        raise ValueError("Parallel workers require read-only existing checkout mode")
    if protected_active_instance_ids and not callable(verify_active_leases):
        raise ValueError("Parallel admission requires a fresh private cleanup verifier")
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
        opened = active_runs(state, protected_active_instance_ids)
        if any(run.get("name") == request["instance_name"] for run in state["runs"]):
            raise ValueError("An attempt name already in the ledger cannot be reused")
        required_resources = {"ssh_key_id"} if local_replica else {"ssh_key_id", "filesystem_id"}
        if not required_resources.issubset(state["resources"]):
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
        verify_instances(instances, opened, request, allow_mixed_profiles=local_replica)
        if stage == "research":
            if not report:
                raise ValueError("Research requires its measured budget and study decision")
            cloud.bind_study(state, report, WORKER_COMMIT)
        checked_at = now()
        hours = (deadline - checked_at) / 3600
        cloud.session_deadline(config, stage, hours, checked_at)
        coverage, reserves = protected_coverage(
            opened, verify_active_leases, now=now, rate=max(rate, config.get("forecast_hourly_rate_usd", rate))
        )
        allowance = cloud.check_allowance(config, state, hours, rate, stage, report, **reserves)
        body = build_body(
            config,
            state,
            armed,
            script,
            report_bytes,
            rate=rate,
            forecast_rate=allowance["forecast_hourly_rate_usd"],
            existing_checkout=existing_checkout,
            normalization_sha256=normalization_sha256,
        )
        creator = cloud.LambdaClient(launch_key)
        # All read-only cloud preflight and payload construction precede this
        # live check. It never dispatches another guard or extends the deadline.
        live = guard.revalidate(guard_path, expected_request=request)
        if live != armed:
            raise ValueError("Armed receipt changed during launch preparation")
        if opened:
            verify_instances(reader.request("GET", "instances"), opened, request, allow_mixed_profiles=local_replica)
            coverage, reserves = protected_coverage(
                opened, verify_active_leases, now=now, rate=allowance["forecast_hourly_rate_usd"]
            )
        started = now()
        hours = (deadline - started) / 3600
        cloud.session_deadline(config, stage, hours, started)
        audited_key = validate_key(launch_key, key_receipt, live, now=started, master_key=master_key)
        allowance = cloud.check_allowance(config, state, hours, rate, stage, report, **reserves)
        pending = {
            "name": request["instance_name"],
            "commit": WORKER_COMMIT,
            "instance_type": config["instance_type"],
            "region": config["region"],
            "workspace_id": request["workspace_id"],
            "filesystem_id": None if local_replica else state["resources"]["filesystem_id"],
            "filesystem_mount_point": None if local_replica else cloud.REMOTE_ROOT,
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
            **({"parallel_admission": {"protected_cleanup": coverage, **reserves}} if coverage else {}),
            **(
                {
                    "storage_mode": "replicated_local",
                    "storage_persistence": "ephemeral",
                    "local_storage_source": "/var/lib/scalar-pi-replica",
                    "local_storage_mount": cloud.REMOTE_ROOT,
                }
                if local_replica
                else {}
            ),
        }
        state["pending_launch"] = pending
        cloud.save_state(state_path, state)
        try:
            # Even suspension after the live check cannot extend admission:
            # the provider rejects this creator key after its actual expiry.
            validate_key(launch_key, key_receipt, live, now=now(), master_key=master_key)
            if coverage and not 0 <= now() - coverage["observed_unix"] <= 30:
                raise ValueError("Protected cleanup verification became stale before launch")
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
