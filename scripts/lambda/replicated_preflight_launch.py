"""One-shot replicated-local host preflight using the existing guarded key lifecycle.

No calls without --execute. The private control root supplies its reviewed
credential/lease verifier. New attempt snapshots never replace current workers.
This bootstrap reserves no scientific assignment and starts no training.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import cloud
from github_guard import GitHubGuard
import guarded
import replicated_bootstrap
from watchdog import WORKER_COMMIT
from watchdog import validate_request

PRIVATE = Path.home() / ".config/scalar-pi"
REPO = Path(__file__).resolve().parents[2]
ID = r"[a-f0-9]{32}"
SHA = r"[a-f0-9]{64}"
WORKER_NAME = "replicated_preflight.sh"
PRIVATE_PINS = {'extra_worker_launch.py': '2fc2ffde7639cf53e2a1b7046c93130780534c24e90d8c3f29ee2163544ecc7e', 'scoped_cleanup_20261116.py': '64cd12c6e6b799bcd7f28d80cea4380e2846ae68f31f722bcdb0b83b0ae72a6f', 'aws_controller_20261116.py': 'ac4de7881b7beb4b5a40c142d792e63882a3bc26f2f835cb52342f593562f471', 'observe_parallel_coverage_concurrent_20261116.py': '8304817781c0584889f278338bfe11402d552d281d0216b1c575996a09ca5aff', 'gh_private.py': '2952cf0a7c1071135a64b7b7d5051a9b8bd413bc184e05cb8b695fc3ca12530a'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def pinned(reference, *, root=PRIVATE):
    require(set(reference) == {"path", "sha256"}, "Pin exact input bytes")
    path = Path(reference["path"])
    require(path.is_absolute() and path.resolve() == path and path.is_relative_to(root)
            and path.is_file() and not path.is_symlink(), "Use canonical private input files")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"], "Pinned input changed")
    return path


def private_helpers(root):
    """Load the existing reviewed Windows credential/coverage helpers explicitly."""
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root, "Canonical control root required")
    require(Path(replicated_bootstrap.__file__).resolve().parent == Path(__file__).resolve().parent,
            "Local storage bootstrap must come from this controller checkout")
    for name, expected in PRIVATE_PINS.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                "Private helper changed: " + name)
    before = list(sys.path)
    try:
        sys.path.append(str(root))
        spec = importlib.util.spec_from_file_location("replica_existing_launch", root / "extra_worker_launch.py")
        existing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(existing)
        require(existing.cloud is cloud and existing.guarded is guarded,
                "Private imports must use this reviewed controller")
    finally:
        sys.path[:] = before
    return existing


def load_plan(path, *, root=PRIVATE, now=time.time):
    scripts = Path(__file__).resolve().parent
    require(all(Path(module.__file__).resolve().parent == scripts
                for module in (cloud, guarded, replicated_bootstrap))
            and Path(validate_request.__globals__["__file__"]).resolve().parent == scripts,
            "Run the portable adapter in a fresh process using this controller checkout")
    plan = read(path)
    require(set(plan) == {"schema_version", "attempt", "controller_commit", "worker_commit", "workspace_id",
                         "initial_worker_purpose", "serialized_github_admission", "admission_policy", "deadline_utc",
                         "config", "authorization", "worker_script", "github_cleanup_receipt", "protected_workers"},
            "Plan must contain only replicated host admission fields, without a measured study or assignment")
    require(plan["schema_version"] == 1, "Unsupported preflight-only plan")
    require(
        re.fullmatch(r"scalar-pi-[a-f0-9]{12}", plan["attempt"]),
        "Use a unique explicit attempt",
    )
    require(
        re.fullmatch(r"[a-f0-9]{40}", plan["controller_commit"]),
        "Pin the published controller commit",
    )
    require(
        plan["worker_commit"] == WORKER_COMMIT, "The actual bootstrap checkout must remain f532"
    )
    require(
        plan["initial_worker_purpose"] == "host_preflight_only",
        "Initial worker cannot dispatch training",
    )
    require(
        plan["serialized_github_admission"] is True,
        "GitHub admission must remain serialized",
    )
    require(
        plan.get("admission_policy") == "replicated_host_preflight_only",
        "Only qualification preflight is admitted",
    )
    require(
        not any(
            key in plan for key in ("study_plan", "budget_report", "selected_study")
        ),
        "Preflight-only admission must not supply a measured study decision",
    )
    paths = {
        name: pinned(plan[name], root=root)
        for name in (
            "config",
            "authorization",
            "worker_script",
            "github_cleanup_receipt",
        )
    }
    require(
        paths["worker_script"].name == WORKER_NAME
        and plan["worker_script"]["sha256"] == hashlib.sha256(Path(__file__).with_name(WORKER_NAME).read_bytes()).hexdigest(),
        "Only the reviewed local-storage host preflight script is admitted",
    )
    config, authorization = (read(paths[name]) for name in ("config", "authorization"))
    cloud.validate_config(config)
    require(cloud.storage_mode(config) == "replicated_local", "Explicit replicated_local mode required")
    require(
        cloud.full_study_authorization(config)["full_study_authorized"] is True,
        "Explicit full-study policy is required",
    )
    require(
        authorization.get("full_study_authorized") is True
        and authorization.get("backup_gpus_authorized") is True,
        "Full-study backup authorization is missing",
    )
    require(
        authorization["seeds"] == [42, 17, 93]
        and authorization["arms"] == ["B0", "B1", "P"]
        and authorization["optimizer_updates_per_arm_seed"] == 30000
        and authorization["evaluation_episodes"] == 40000,
        "Authorization differs from the matched study",
    )
    request = {
        "instance_name": plan["attempt"],
        "workspace_id": plan["workspace_id"],
        "worker_commit": WORKER_COMMIT,
        "region": config["region"],
        "instance_type": config["instance_type"],
        "storage_mode": "replicated_local",
        "deadline_utc": plan["deadline_utc"],
        "launch_window_seconds": 900,
    }
    request, deadline = validate_request(request, now())
    cloud.session_deadline(config, "qualification", (deadline - now()) / 3600, now())
    bindings = plan["protected_workers"]
    require(
        isinstance(bindings, list) and bindings,
        "List every existing protected worker explicitly",
    )
    require(
        len({item["instance_id"] for item in bindings}) == len(bindings),
        "Protected worker IDs must be unique",
    )
    for item in bindings:
        require(
            set(item) == {"instance_id", "launch_receipt", "lease_metadata"}
            and re.fullmatch(ID, item["instance_id"]),
            "Invalid protected worker binding",
        )
        pinned(item["launch_receipt"], root=root)
        pinned(item["lease_metadata"], root=root)
    cleanup = read(paths["github_cleanup_receipt"])
    require(
        cleanup["status"] == "installed_in_github_environment",
        "GitHub cleanup credential is not installed",
    )
    expiry = dt.datetime.fromisoformat(cleanup["expires_at"]).timestamp()
    require(
        expiry > deadline + 1500,
        "GitHub cleanup key expires before this complete lease",
    )
    return plan, paths, config, None, request


def launch(plan_path, *, root=PRIVATE, reader, guard, verifier, helpers=None, now=time.time):
    plan, paths, config, _budget, request = load_plan(plan_path, root=root, now=now)
    helpers = private_helpers(root) if helpers is None else helpers
    plan_digest = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    require(read(plan_path) == plan, "Plan changed during validation")
    folder = root / "extra-workers" / plan["attempt"]
    ids = tuple(item["instance_id"] for item in plan["protected_workers"])
    state_path = root / "lambda-state.json"
    with cloud.project_lock(root / "extra-worker-admission.lock"):
        require(
            not folder.exists(),
            "Attempt already recorded; reconcile it instead of redispatching",
        )
        state = cloud.load_state(state_path)
        opened = guarded.active_runs(state, ids)
        rate = cloud.select_type(reader, config)
        guarded.verify_instances(reader.request("GET", "instances"), opened, request, allow_mixed_profiles=True)
        _, reserves = guarded.protected_coverage(
            opened,
            verifier,
            now=now,
            rate=max(rate, config.get("forecast_hourly_rate_usd", rate)),
        )
        deadline = dt.datetime.fromisoformat(request["deadline_utc"]).timestamp()
        allowance = cloud.check_allowance(
            config,
            state,
            (deadline - now()) / 3600,
            rate,
            "qualification",
            None,
            **reserves,
        )
        service = read(root / "cloud-service-identities-20261116.json")["identities"][
            "launcher"
        ]
        identity = service["identity"]["id"]
        require(
            service["membership"]["workspace_id"] == request["workspace_id"],
            "Launcher workspace differs",
        )
        assignments, roles = helpers.audit_identity(
            reader, identity, request["workspace_id"], [guarded.LAUNCH_ROLE_ID]
        )
        role = roles[0]
        require(
            role["name"] == "workspace-member"
            and "compute:instance:launch" in role["permission_ids"],
            "Launcher role cannot launch",
        )
        helpers.verify_github_key(
            reader,
            read(paths["github_cleanup_receipt"]),
            request["workspace_id"],
            deadline,
        )
        folder.mkdir(parents=True, exist_ok=False)
        # Waiting for GitHub must not allow draft edits to change worker inputs.
        for name in (
            "config",
            "authorization",
            "worker_script",
            "github_cleanup_receipt",
        ):
            payload = paths[name].read_bytes()
            require(
                hashlib.sha256(payload).hexdigest() == plan[name]["sha256"],
                "Pinned input changed before snapshot",
            )
            destination = folder / (name + paths[name].suffix)
            with destination.open("xb") as stream:
                stream.write(payload)
            paths[name] = destination
        require(
            hashlib.sha256(Path(plan_path).read_bytes()).hexdigest() == plan_digest,
            "Plan bytes changed before dispatch",
        )
        with (folder / "plan.json").open("xb") as stream:
            stream.write(Path(plan_path).read_bytes())
        request_path = folder / "request.json"
        cloud.save_state(request_path, request)
        receipt_path = folder / "launch.json"
        receipt = {
            "schema_version": 1,
            "status": "dispatch_pending",
            "attempt": plan["attempt"],
            "controller_commit": plan["controller_commit"],
            "worker_commit": WORKER_COMMIT,
            "request_path": str(request_path),
            "config_sha256": plan["config"]["sha256"],
            "worker_script_sha256": plan["worker_script"]["sha256"],
            "launch_plan_sha256": plan_digest,
            "allowance_preflight": allowance,
            "initial_worker_purpose": "host_preflight_only",
            "admission_policy": "replicated_host_preflight_only",
            "stage": "qualification",
            "measured_study_report": None,
        }
        cloud.save_state(receipt_path, receipt)
        guard_path = guard.dispatch(request, dry_run=False)
        receipt.update(status="arming", guard_path=str(guard_path))
        cloud.save_state(receipt_path, receipt)
        armed = guard.wait_ready(guard_path, timeout_seconds=600)
        expiry_seconds = min(600, int(armed["launch_not_after_unix"] - now()) - 30)
        require(expiry_seconds > 120, "Launch window cannot fit a new creator key")
        key_path = folder / "launcher-key-receipt.json"
        cloud.save_state(
            key_path,
            {
                "status": "creation_pending",
                "identity_id": identity,
                "recorded_unix": now(),
            },
        )
        created = reader.request(
            "POST",
            f"identities/{identity}/api-keys",
            {
                "name": "scalar-pi launch " + plan["attempt"][-12:],
                "expires_in": expiry_seconds,
            },
        )
        try:
            key = created.pop("api_key")
            key_receipt = {
                "schema_version": 1,
                "identity_id": identity,
                "key_id": created["id"],
                "expires_at": created["expires_at"],
                "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "workspace_id": request["workspace_id"],
                "workspace_ids": [request["workspace_id"]],
                "role_ids": [role["id"]],
                "role_names": [role["name"]],
                "permission_ids": role["permission_ids"],
                "account_role_ids": [],
                "inherited_role_ids": [],
                "group_ids": [],
                "audited_assignments": assignments,
                "audited_unix": now(),
            }
            cloud.save_state(key_path, key_receipt)
            run = guarded.launch(
                config,
                state_path,
                paths["worker_script"],
                key_path,
                guard_path,
                request,
                guard,
                reader=reader,
                launch_key=key,
                stage="qualification",
                budget_report_path=None,
                protected_active_instance_ids=ids,
                verify_active_leases=verifier,
                existing_checkout=False,
                now=now,
            )
            receipt.update(
                status="launched",
                instance_id=run["instance_id"],
                watchdog=run["watchdog"],
                deadline_unix=run["deadline_unix"],
                deadline_utc=request["deadline_utc"],
                watchdog_url="https://github.com/xyleus1/scalar-pi/actions/runs/"
                + armed["github"]["run_id"],
            )
            cloud.save_state(receipt_path, receipt)
            return receipt
        finally:
            reader.request("DELETE", "api-keys/" + created["id"])
            cloud.save_state(
                folder / "launcher-key-revoked.json",
                {"key_id": created["id"], "observed_unix": now()},
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, default=PRIVATE)
    parser.add_argument("--aws-plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.control_root.resolve()
    plan, _, _, _, _ = load_plan(args.plan, root=root)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "prepared_no_calls",
                    "attempt": plan["attempt"],
                    "storage_mode": "replicated_local",
                }
            )
        )
        return
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        == plan["controller_commit"],
        "Local controller commit differs",
    )
    require(
        not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        ).strip(),
        "Controller checkout must be clean",
    )
    helpers = private_helpers(root)
    # Trusted private sibling modules remain behind the active controller path.
    before = list(sys.path)
    try:
        sys.path.append(str(root))
        from aws_controller_20261116 import clients
        from gh_private import GH
        from gh_private import environment
        from observe_parallel_coverage_concurrent_20261116 import concurrent_verifier
        from scoped_cleanup_20261116 import adapter
        from scoped_cleanup_20261116 import require_qualified

        require_qualified(args.aws_plan)
        aws = clients(args.aws_plan)
        context = adapter(args.aws_plan)
    finally:
        # Old private imports insert their own repository path. Restore the
        # published controller's precedence before creating a launch payload.
        sys.path[:] = before
        sys.path.append(str(root))
    os.environ["GH_TOKEN"] = environment()["GH_TOKEN"]
    reader = cloud.LambdaClient(os.environ.get("LAMBDA_API_KEY"))
    guard = GitHubGuard(GH, root / "github-guards", plan["controller_commit"])
    verifier = concurrent_verifier(plan["protected_workers"], aws, reader,
                                  root=root, now=time.time, context_loader=context.load_context)
    result = launch(args.plan, root=root, reader=reader, guard=guard, verifier=verifier, helpers=helpers)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "attempt",
                    "instance_id",
                    "deadline_unix",
                    "watchdog_url",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
