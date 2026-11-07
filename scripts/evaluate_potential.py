"""Plan or execute the preregistered paired LIBERO matrix (dry-run by default).

The simulator and server intentionally run in separate Python environments.
Set command prefixes in the JSON manifest; subprocesses never use a shell.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.research import evaluation


def _validate_suite_log(manifest: dict, checkpoint_id: str, steps: int, suite: str, path: str) -> dict:
    """Use a diagnostic validation *view*, without relaxing the original matrix."""
    variant = next(v for v in manifest["variants"] if v["checkpoint_id"] == checkpoint_id)
    subset = {**manifest, "diagnostic": True, "variants": [variant], "num_steps": [steps], "suites": [suite]}
    try:
        return evaluation.validate_results(
            evaluation.read_episodes([path]),
            subset,
            require_primary_checkpoint=not manifest.get("diagnostic", False),
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Existing suite log needs investigation; cannot resume {path}: {exc}") from exc


def commands(manifest: dict, *, resume_completed_suites: bool = False) -> list[dict]:
    """Return one server invocation and its sequential suite clients per setting."""
    cells = evaluation.matrix_cells(manifest)
    server_prefix = manifest.get(
        "server_command_prefix", ["uv", "run", "--no-sync", "python", "scripts/serve_policy.py"]
    )
    client_prefix = manifest.get(
        "client_command_prefix", ["examples/libero/.venv/bin/python", "examples/libero/main.py"]
    )
    for prefix in (server_prefix, client_prefix):
        if not isinstance(prefix, list) or not prefix or any(not isinstance(part, str) for part in prefix):
            raise ValueError("Command prefixes must be nonempty JSON lists of argv strings")
    port = int(manifest.get("port", 8000))
    batches = {}
    for cell in cells:
        key = (cell["checkpoint_id"], cell["num_steps"])
        if key not in batches:
            server = [
                *server_prefix,
                "--port",
                str(port),
                "--num-steps",
                str(cell["num_steps"]),
                "--checkpoint-id",
                cell["checkpoint_id"],
                "policy:checkpoint",
                "--policy.config",
                cell["config"],
                "--policy.dir",
                cell["checkpoint_dir"],
            ]
            batches[key] = {
                "checkpoint_id": key[0],
                "num_steps": key[1],
                "server": server,
                "clients": [],
                "completed_suites": [],
            }
        if Path(cell["episode_log_path"]).exists():
            if not resume_completed_suites:
                raise FileExistsError(
                    f"Existing suite log {cell['episode_log_path']}; use --resume-completed-suites to validate and skip"
                )
            validation = _validate_suite_log(manifest, key[0], key[1], cell["suite"], cell["episode_log_path"])
            batches[key]["completed_suites"].append(
                {"suite": cell["suite"], "log": cell["episode_log_path"], **validation}
            )
            continue
        client = [
            *client_prefix,
            "--args.host",
            "127.0.0.1",
            "--args.port",
            str(port),
            "--args.task-suite-name",
            cell["suite"],
            "--args.num-trials-per-task",
            str(manifest.get("num_trials_per_task", 50)),
            "--args.seed",
            str(manifest.get("rollout_seed", 7)),
            "--args.paired-noise-seed",
            str(manifest.get("paired_noise_seed", 7)),
            "--args.episode-log-path",
            cell["episode_log_path"],
            "--args.checkpoint-id",
            cell["checkpoint_id"],
            "--args.checkpoint-dir",
            cell["checkpoint_dir"],
            "--args.config",
            cell["config"],
            "--args.group",
            cell["group"],
            "--args.num-steps",
            str(cell["num_steps"]),
            "--args.video-out-path",
            str(Path(cell["episode_log_path"]).with_suffix("")),
        ]
        if cell.get("reference", False):
            client.append("--args.reference")
        else:
            client.extend(["--args.training-seed", str(cell["training_seed"])])
        if not manifest.get("save_videos", True):
            client.append("--args.no-save-videos")
        if manifest.get("num_tasks_per_suite", 10) != 10:
            client.extend(["--args.max-tasks", str(manifest["num_tasks_per_suite"])])
        batches[key]["clients"].append({"suite": cell["suite"], "argv": client, "log": cell["episode_log_path"]})
    for batch in batches.values():
        batch["status"] = "pending" if batch["clients"] else "complete"
    return list(batches.values())


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Policy server exited with status {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Policy server was not ready after {timeout} seconds")


def execute(manifest: dict, *, resume_completed_suites: bool = False) -> None:
    root = Path(__file__).resolve().parents[1]
    port = int(manifest.get("port", 8000))
    # Recheck disk state at execution time, including logs validated in the dry-run plan.
    batches = commands(manifest, resume_completed_suites=resume_completed_suites)
    pending_batches = [batch for batch in batches if batch["clients"]]
    pending_ids = {batch["checkpoint_id"] for batch in pending_batches}
    if any(
        v["checkpoint_id"] in pending_ids and v["checkpoint_dir"].startswith("REPLACE_WITH_")
        for v in manifest["variants"]
    ):
        raise ValueError("Replace example checkpoint paths before executing the matrix")
    # Do not accidentally evaluate a stale or unrelated policy server.
    if pending_batches:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                raise RuntimeError(f"Port {port} is already occupied; choose a free port")
        except (ConnectionRefusedError, TimeoutError):
            pass
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for batch in pending_batches:
        server_log = (
            Path(manifest.get("output_dir", "results/potential"))
            / batch["checkpoint_id"]
            / f"k{batch['num_steps']}"
            / "server.log"
        )
        server_log.parent.mkdir(parents=True, exist_ok=True)
        with server_log.open("a", encoding="utf-8") as stream:
            stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            stream.write(f"\n=== server attempt {stamp}: {batch['checkpoint_id']} K={batch['num_steps']} ===\n")
            stream.flush()
            server = subprocess.Popen(batch["server"], cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                _wait_for_server(server, port, float(manifest.get("server_start_timeout_seconds", 900)))
                for client in batch["clients"]:
                    if server.poll() is not None:
                        raise RuntimeError("Policy server exited before the next suite")
                    subprocess.run(
                        client["argv"],
                        cwd=root,
                        env=env,
                        check=True,
                        timeout=float(manifest.get("suite_timeout_seconds", 86400)),
                    )
                    _validate_suite_log(
                        manifest, batch["checkpoint_id"], batch["num_steps"], client["suite"], client["log"]
                    )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    records = evaluation.read_episodes([cell["episode_log_path"] for cell in evaluation.matrix_cells(manifest)])
    print(json.dumps(evaluation.validate_results(records, manifest), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="Launch the matrix; default only prints argv commands")
    parser.add_argument(
        "--resume-completed-suites",
        action="store_true",
        help="Validate and skip complete existing suite logs; reject partial or invalid logs",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # Paths in commands are relative to the repository, independently of the caller's cwd.
    output_dir = Path(manifest.get("output_dir", "results/potential"))
    if not output_dir.is_absolute():
        manifest["output_dir"] = str(Path(__file__).resolve().parents[1] / output_dir)
    batches = commands(manifest, resume_completed_suites=args.resume_completed_suites)
    total = (
        len(evaluation.matrix_cells(manifest))
        * manifest.get("num_trials_per_task", 50)
        * manifest.get("num_tasks_per_suite", 10)
    )
    print(
        json.dumps(
            {
                "dry_run": not args.execute,
                "resume_completed_suites": args.resume_completed_suites,
                "expected_episodes": total,
                "pending_suites": sum(len(batch["clients"]) for batch in batches),
                "completed_suites": sum(len(batch["completed_suites"]) for batch in batches),
                "batches": batches,
            },
            indent=2,
        )
    )
    if args.execute:
        execute(manifest, resume_completed_suites=args.resume_completed_suites)


if __name__ == "__main__":
    main()
