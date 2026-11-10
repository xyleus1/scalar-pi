"""Preview or run component timing for the full LIBERO cost envelope.

Uses the existing policy benchmark, native LIBERO runner and retained smoke logs.
No timing diagnostic is a benchmark success-rate observation. --execute is required.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.research import evaluation as ev

GROUPS = {"B0": "pi05_libero_b0", "B1": "pi05_libero_b1", "P": "pi05_libero_potential", "reference": "pi05_libero"}


def positive(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero)
    ):
        raise ValueError(f"Invalid measured {name}")
    return value


def commands(spec):
    policies = spec.get("policies", [])
    if len(policies) != 4 or {item.get("group") for item in policies} != set(GROUPS):
        raise ValueError("Calibration requires exactly B0/B1/P/reference")
    output = Path(spec["output_dir"])
    jobs = []
    for policy in policies:
        group = policy["group"]
        if policy.get("config") != GROUPS[group]:
            raise ValueError("Calibration config differs from its declared architecture")
        prefix = policy.get("command_prefix", spec.get("policy_command_prefix"))
        for steps in (1, 10):
            path = output / f"{group}-k{steps}.json"
            jobs.append(
                {
                    "kind": "policy",
                    "group": group,
                    "num_steps": steps,
                    "path": str(path),
                    "checkpoint_dir": policy["checkpoint_dir"],
                    "argv": [
                        *(prefix or []),
                        "--config",
                        policy["config"],
                        "--checkpoint",
                        policy["checkpoint_dir"],
                        "--corpus",
                        spec["corpus"],
                        "--num-steps",
                        str(steps),
                        "--report",
                        str(path),
                    ],
                }
            )
    for suite in ev.SUITES:
        path = output / f"{suite}.json"
        jobs.append(
            {
                "kind": "scene",
                "suite": suite,
                "path": str(path),
                "argv": [
                    *spec.get("simulator_command_prefix", []),
                    "--args.task-suite-name",
                    suite,
                    "--args.calibration-output",
                    str(path),
                    "--args.no-save-videos",
                ],
            }
        )
    for job in jobs:
        if (
            not job["argv"]
            or job["argv"][0].startswith("--")
            or any(not isinstance(value, str) for value in job["argv"])
        ):
            raise ValueError("Calibration commands require an explicit argv prefix and string paths")
    return jobs


def checkpoint_hash(path):
    path = Path(path)
    if (path / "model.safetensors").is_file():
        return ev.file_digest(path / "model.safetensors")
    files = {
        item.relative_to(path).as_posix(): ev.file_digest(item)
        for item in sorted((path / "params").rglob("*"))
        if item.is_file()
    }
    if not files:
        raise ValueError("Calibration checkpoint has no staged weights")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def smoke_overhead(path):
    manifest = json.loads(Path(path).read_text())
    if not manifest.get("diagnostic") or set(manifest.get("suites", [])) != set(ev.SUITES):
        raise ValueError("Transport timing requires the four-suite diagnostic smoke manifest")
    paths = [cell["episode_log_path"] for cell in ev.matrix_cells(manifest)]
    rows = ev.read_episodes(paths)
    ev.validate_results(rows, manifest)
    if any(row["server_metadata"].get("runtime", {}).get("backend") != "jax" for row in rows):
        raise ValueError("Transport evidence must come from the native released-reference smoke")
    overhead = (
        max(max(0, sample["client_roundtrip_ms"] - sample["total_ms"]) for row in rows for sample in row["latencies"])
        / 1000
    )
    return overhead, [str(path), *paths, *(str(Path(path).with_suffix(".provenance.json")) for path in paths)]


def compose(spec, jobs, *, source, model_image, simulator_image, corpus_hash, checkpoint_hashes):
    """Conservative native-step envelope; outcomes and early-success lengths are unused."""
    overhead, evidence_paths = smoke_overhead(spec["smoke_manifest"])
    policies, scenes = {}, {}
    for job in jobs:
        path = Path(job["path"])
        report = json.loads(path.read_text())
        execution_path = path.with_suffix(".execution.json")
        execution = json.loads(execution_path.read_text())
        if execution.get("report_sha256") != ev.file_digest(path) or execution.get("argv") != job["argv"]:
            raise ValueError("Calibration command/report receipt differs from the selected job")
        elapsed = positive(execution.get("seconds"), "command duration")
        evidence_paths.extend((str(path), str(execution_path)))
        if job["kind"] == "policy":
            group, steps = job["group"], job["num_steps"]
            policy = report.get("policy", {})
            if (
                report.get("purpose") != "isolated_warmed_policy_latency"
                or report.get("source") != source
                or report.get("corpus_sha256") != corpus_hash
                or report.get("num_steps") != steps
                or report.get("measured_calls", 0) < 100
                or report.get("warmup_calls", 0) < 10
                or report.get("corpus_observations") != 40
                or report.get("batch_size") != 1
                or policy.get("config_name") != GROUPS[group]
                or policy.get("num_steps") != steps
                or policy.get("checkpoint_sha256") != checkpoint_hashes[group]
                or policy.get("runtime", {}).get("container_image") != model_image
                or policy.get("runtime", {}).get("backend") != ("jax" if group == "reference" else "pytorch")
            ):
                raise ValueError("Policy calibration identity or native settings differ")
            calls = report.get("calls", [])
            if len(calls) != report["measured_calls"] or {sample.get("corpus_index") for sample in calls} != set(
                range(40)
            ):
                raise ValueError("Policy timing does not cover all forty audited observations")
            values = [positive(sample.get("total_ms"), "policy total_ms") / 1000 for sample in calls]
            cold = positive(report.get("startup", {}).get("first_query_seconds"), "first-query duration")
            loading = positive(report.get("startup", {}).get("model_load_seconds"), "model-load duration")
            # Retain process imports, model loading and all warmup/compilation in startup.
            startup = max(cold + loading, elapsed - sum(values))
            policies[(group, steps)] = (ev._percentile(values, 0.95) + overhead, startup)  # noqa: SLF001
        else:
            suite = job["suite"]
            provenance = report.get("provenance", {})
            tasks = report.get("tasks", [])
            if (
                report.get("purpose") != "libero_scene_timing"
                or report.get("suite") != suite
                or report.get("render_size") != 256
                or report.get("resize_size") != 224
                or report.get("num_steps_wait") != 10
                or report.get("max_steps") != ev.MAX_STEPS[suite]
                or provenance.get("source") != source
                or provenance.get("container_image") != simulator_image
                or len(tasks) != 10
                or sorted(task.get("task_id") for task in tasks) != list(range(10))
            ):
                raise ValueError("Scene calibration identity or complete native coverage differs")
            estimates = []
            for task in tasks:
                if task.get("task_start_parity") is not True:
                    raise ValueError("Task-boundary initialization parity must pass in every scene")
                if task.get("measured_steps") != ev.MAX_STEPS[suite] + 10:
                    raise ValueError("Scene calibration must measure the complete native horizon")
                creation, reset, settle, step, preprocess = (
                    positive(task.get(key), key, zero=True)
                    for key in (
                        "create_seconds",
                        "reset_seconds",
                        "settling_seconds",
                        "step_seconds_p95",
                        "preprocess_seconds_p95",
                    )
                )
                estimates.append((creation, reset + settle + ev.MAX_STEPS[suite] * (step + preprocess)))
            # Conservatively include one whole simulator-process overhead per task restart.
            process_overhead = max(
                0, elapsed - positive(report.get("process_seconds"), "simulator process_seconds")
            ) + positive(report.get("provenance_seconds"), "simulator provenance_seconds", zero=True)
            scenes[suite] = (
                max(value[0] for value in estimates) + process_overhead,
                max(value[1] for value in estimates),
            )
    durations, task_budgets = {}, {}
    for group in GROUPS:
        durations[group], task_budgets[group] = {}, {}
        for steps in (1, 10):
            latency, startup = policies[(group, steps)]
            durations[group][str(steps)], task_budgets[group][str(steps)] = {}, {}
            for suite in ev.SUITES:
                creation, nonpolicy = scenes[suite]
                task_seconds = creation + startup + 50 * (nonpolicy + math.ceil(ev.MAX_STEPS[suite] / 5) * latency)
                durations[group][str(steps)][suite] = task_seconds / 50
                task_budgets[group][str(steps)][suite] = 1.2 * task_seconds
    return {
        "schema_version": 1,
        "purpose": "evaluation_cost_calibration",
        "passed": True,
        "valid_benchmark_evaluation": False,
        "source": source,
        "model_image": model_image,
        "simulator_image": simulator_image,
        "corpus_sha256": corpus_hash,
        "checkpoint_sha256": checkpoint_hashes,
        "evaluation_seconds_per_episode": durations,
        "task_budget_seconds": task_budgets,
        "server_start_budget_seconds": 1.2 * max(value[1] for value in policies.values()),
        "transport_seconds_per_query": overhead,
        "evidence": {path: ev.file_digest(path) for path in sorted(set(evidence_paths))},
        "method": "Native maximum step/query counts, per-scene p95 component timings, warmed policy p95; cold costs charged per task conservatively",
        "limitations": "Measured conservative estimate, not a hard wall-time guarantee; task admission adds20% and no success lengths are used",
    }


def run_job(argv, log, timeout):
    process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
    try:
        status = process.wait(timeout=timeout)
        if status:
            raise subprocess.CalledProcessError(status, argv)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse complete component receipts")
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    jobs = commands(spec)
    if not args.execute:
        print(json.dumps({"dry_run": True, "jobs": jobs}, indent=2))
        return
    if args.output.exists():
        raise FileExistsError("Use a new aggregate timing report path")
    source = ev.source_identity(Path(__file__).resolve().parents[1])
    model_image, simulator_image = os.environ.get("SCALAR_PI_MODEL_IMAGE"), os.environ.get("SCALAR_PI_SIMULATOR_IMAGE")
    if source["tracked_changes"] or not model_image or not simulator_image:
        raise ValueError("Calibrate only the clean pinned source with both actual container identities")
    weights = {policy["group"]: checkpoint_hash(policy["checkpoint_dir"]) for policy in spec["policies"]}
    corpus_hash = ev.file_digest(spec["corpus"])
    smoke_overhead(spec["smoke_manifest"])  # Fail before paid diagnostics if transport evidence is missing.
    deadline = float(os.environ.get("SCALAR_PI_DEADLINE_UNIX") or "inf")
    if math.isnan(deadline) or deadline <= 0:
        raise ValueError("Invalid absolute calibration deadline")
    for job in jobs:
        path = Path(job["path"])
        if path.exists():
            if not args.resume or not path.with_suffix(".execution.json").exists():
                raise FileExistsError(f"Existing calibration evidence needs a complete receipt and --resume: {path}")
            report = json.loads(path.read_text())
            receipt = json.loads(path.with_suffix(".execution.json").read_text())
            if receipt.get("argv") != job["argv"] or receipt.get("report_sha256") != ev.file_digest(path):
                raise ValueError("Existing calibration receipt does not match the selected job")
            if job["kind"] == "policy":
                if (
                    report.get("source") != source
                    or report.get("corpus_sha256") != corpus_hash
                    or report.get("policy", {}).get("checkpoint_sha256") != weights[job["group"]]
                    or report.get("policy", {}).get("runtime", {}).get("container_image") != model_image
                ):
                    raise ValueError("Existing policy timing differs from current inputs")
            elif (
                report.get("provenance", {}).get("source") != source
                or report.get("provenance", {}).get("container_image") != simulator_image
            ):
                raise ValueError("Existing scene timing differs from current inputs")
            continue
        remaining = deadline - time.time() - 600
        if remaining <= 0:
            raise TimeoutError("No calibration time remains before the shutdown reserve")
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with path.with_suffix(".log").open("x") as log:
            run_job(job["argv"], log, remaining if math.isfinite(remaining) else None)
        path.with_suffix(".execution.json").write_text(
            json.dumps(
                {"argv": job["argv"], "seconds": time.monotonic() - started, "report_sha256": ev.file_digest(path)},
                indent=2,
            )
            + "\n"
        )
    report = compose(
        spec,
        jobs,
        source=source,
        model_image=model_image,
        simulator_image=simulator_image,
        corpus_hash=corpus_hash,
        checkpoint_hashes=weights,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Wrote conservative evaluation timing: {args.output}")


if __name__ == "__main__":
    main()
