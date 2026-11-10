"""Inspect research prerequisites without downloading data or provisioning GPUs.

This is an environment/asset audit, not a substitute for a real-model GPU smoke
test. Optional budget estimates consume an explicitly identified measurement
record. Synthetic and CPU measurements never qualify as a measured GPU profile.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys

BASE_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
REQUIRED_PACKAGES = {"torch": "2.7.1", "transformers": "4.53.2"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def positive_number(value, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def estimate_budget(profile: dict, hourly_rate_usd: float, budget_usd: float) -> dict:
    """Forecast cost from a complete, explicitly measured full-model GPU record.

    The caller supplies the current provider price; no price is hard-coded.
    This function validates a record's declarations, not the truth of external
    measurements. Preserve the underlying profiler logs alongside this record.
    """
    if profile.get("synthetic_input") is True or profile.get("valid_training_cost_profile") is False:
        raise ValueError("Synthetic correctness diagnostics are not valid training cost profiles")
    if profile.get("measured_gpu") is not True or profile.get("model_kind") != "pi05_full":
        raise ValueError("Budget qualification requires measured_gpu=true and model_kind='pi05_full'")
    if not isinstance(profile.get("hardware"), str) or not profile["hardware"].strip():
        raise ValueError("hardware must identify the measured GPU")
    if not isinstance(profile.get("measurement_source"), str) or not profile["measurement_source"].strip():
        raise ValueError("measurement_source must identify the retained profiler logs")
    for key in ("warmup_optimizer_updates", "timed_optimizer_updates", "training_seeds", "evaluation_episodes"):
        if isinstance(profile.get(key), bool) or not isinstance(profile.get(key), int):
            raise ValueError(f"{key} must be an integer")
        positive_number(profile[key], key)
    measured_updates = profile["timed_optimizer_updates"]
    if measured_updates < 100:
        raise ValueError("At least 100 warmed complete optimizer updates are required")
    if profile.get("effective_batch_size") != 256 or profile.get("optimizer_updates") != 30_000:
        raise ValueError("The reference protocol requires effective_batch_size=256 and optimizer_updates=30000")
    times = profile.get("seconds_per_optimizer_update")
    if not isinstance(times, dict) or not times:
        raise ValueError("seconds_per_optimizer_update must contain a measured value for each planned variant")
    seconds = sum(positive_number(value, f"seconds_per_optimizer_update.{key}") for key, value in times.items())
    seeds = positive_number(profile.get("training_seeds"), "training_seeds")
    episodes = positive_number(profile.get("evaluation_episodes"), "evaluation_episodes")
    evaluation_cells = profile.get("evaluation_cells")
    if evaluation_cells is not None:
        if not isinstance(evaluation_cells, list) or not evaluation_cells:
            raise ValueError("evaluation_cells must contain measured per-group/K/suite durations")
        measured_episodes = 0
        evaluation_seconds = 0.0
        for cell in evaluation_cells:
            count = cell.get("episodes")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Every evaluation cell needs a positive integer episode count")
            measured_episodes += count
            evaluation_seconds += count * positive_number(cell.get("seconds_per_episode"), "seconds_per_episode")
        if measured_episodes != episodes:
            raise ValueError("Measured evaluation cells do not cover the declared episode count")
        evaluation_hours = evaluation_seconds / 3600
    else:
        throughput = positive_number(profile.get("episodes_per_gpu_hour"), "episodes_per_gpu_hour")
        evaluation_hours = episodes / throughput
    fixed_cost = positive_number(profile.get("fixed_cost_usd", 0), "fixed_cost_usd", allow_zero=True)
    rate = positive_number(hourly_rate_usd, "hourly_rate_usd")
    ceiling = positive_number(budget_usd, "budget_usd")
    train_hours = seeds * 30_000 * seconds / 3600
    gpu_cost = (train_hours + evaluation_hours) * rate
    total = 1.2 * gpu_cost + fixed_cost
    return {
        "basis": "declared full-model GPU measurements; retain and audit the source logs",
        "hardware": profile["hardware"],
        "measurement_source": profile["measurement_source"],
        "variants": sorted(times),
        "training_gpu_hours": train_hours,
        "evaluation_gpu_hours": evaluation_hours,
        "hourly_rate_usd": rate,
        "gpu_cost_before_headroom_usd": gpu_cost,
        "headroom_fraction": 0.2,
        "fixed_cost_usd": fixed_cost,
        "projected_total_usd": total,
        "budget_usd": ceiling,
        "within_budget": total <= ceiling,
    }


def _study_progress(profile: dict) -> tuple[dict, set, list]:
    """Only subtract work backed by complete checkpoint or validated suite artifacts."""
    identity = profile.get("experiment_identity", {})
    if not isinstance(identity, dict) or set(identity) - {"code_commit", "container_image_id"}:
        raise ValueError("experiment_identity supports code_commit and container_image_id")
    for key, pattern in (("code_commit", r"[0-9a-f]{40}"), ("container_image_id", r"sha256:[0-9a-f]{64}")):
        if key in identity and (not isinstance(identity[key], str) or not re.fullmatch(pattern, identity[key])):
            raise ValueError(f"Invalid experiment_identity.{key}")

    def check_identity(record):
        if any(record.get(key) != value for key, value in identity.items()):
            raise ValueError("Completed artifact belongs to a different experiment identity")

    training, evaluation_cells, receipts = {}, set(), []
    for field in ("completed_training", "completed_evaluation"):
        if not isinstance(profile.get(field, []), list):
            raise ValueError(f"{field} must be a list of artifact-backed records")
    for record in profile.get("completed_training", []):
        if not isinstance(record, dict):
            raise ValueError("Completed training record must be an object")
        group, seed, updates = (record.get(key) for key in ("group", "training_seed", "optimizer_updates"))
        if group not in {"B0", "B1", "P"} or type(seed) is not int or seed not in {17, 42, 93}:
            raise ValueError("Completed training has an unsupported group or seed")
        if type(updates) is not int or not 1 <= updates <= 30_000:
            raise ValueError("Completed optimizer_updates must be an integer in 1..30000")
        if (group, seed) in training:
            raise ValueError("Duplicate completed training run; declare only its latest saved checkpoint")
        if not isinstance(record.get("checkpoint_dir"), str) or not record["checkpoint_dir"]:
            raise ValueError("Completed training requires checkpoint_dir")
        directory = Path(record["checkpoint_dir"])
        metadata = read_json(directory / "potential_metadata.json")
        expected = {
            "schema_version": 1,
            "trainer": "potential_pi",
            "run_mode": "training",
            "training_seed": seed,
            "optimizer_updates": updates,
            "freeze_vlm": True,
            "action_horizon": 10,
            "action_dim": 32,
            "inference_weights": "ema",
            "velocity_mode": "potential_dot" if group == "P" else "direct",
            "initialization": "pretrained_output" if group == "B0" else "zero_output",
        }
        if any(
            type(metadata.get(key)) is not type(value) or metadata.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Completed training declaration disagrees with its research checkpoint")
        check_identity(metadata)
        # Check actual saved bytes; a counter or a path label alone cannot reduce a budget.
        for filename, key in (("model.safetensors", "model_sha256"), ("raw_model.safetensors", "raw_model_sha256")):
            if not metadata.get(key) or sha256_file(directory / filename) != metadata[key]:
                raise ValueError("Completed checkpoint weight hash mismatch")
        state_path = directory / "training_state.pt"
        if not state_path.is_file() or state_path.stat().st_size == 0:
            raise ValueError("Completed training checkpoint lacks resumable state")
        training[group, seed] = updates
        receipts.append({**record, "metadata_sha256": sha256_file(directory / "potential_metadata.json")})
    for record in profile.get("completed_evaluation", []):
        if not isinstance(record, dict):
            raise ValueError("Completed evaluation record must be an object")
        group, seed, steps, suite = (record.get(key) for key in ("group", "training_seed", "num_steps", "suite"))
        valid_seed = seed is None if group == "reference" else type(seed) is int and seed in {17, 42, 93}
        if (
            group not in {"B0", "B1", "P", "reference"}
            or not valid_seed
            or type(steps) is not int
            or steps not in (1, 10)
        ):
            raise ValueError("Completed evaluation has an unsupported group, seed, or solver")
        if suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError("Completed evaluation has an unsupported suite")
        key = (group, seed, steps, suite)
        if key in evaluation_cells:
            raise ValueError("Duplicate completed evaluation cell")
        if not isinstance(record.get("manifest_path"), str) or not record["manifest_path"]:
            raise ValueError("Completed evaluation requires its original manifest_path")
        from openpi.research import evaluation

        manifest = read_json(Path(record["manifest_path"]))
        if manifest.get("diagnostic", False) or not manifest.get("require_provenance", True):
            raise ValueError("Diagnostic evaluations cannot reduce the benchmark budget")
        matches = [
            cell
            for cell in evaluation.matrix_cells(manifest)
            if (cell["group"], cell.get("training_seed"), cell["num_steps"], cell["suite"]) == key
        ]
        if len(matches) != 1:
            raise ValueError("Completed evaluation cell is absent or ambiguous in its manifest")
        cell = matches[0]
        attempt_key = f"{cell['checkpoint_id']}/k{steps}/{suite}"
        variant = next(v for v in manifest["variants"] if v["checkpoint_id"] == cell["checkpoint_id"])
        subset = {
            **manifest,
            "diagnostic": True,
            "require_provenance": True,
            "variants": [variant],
            "num_steps": [steps],
            "suites": [suite],
            "suite_attempts": {k: v for k, v in manifest.get("suite_attempts", {}).items() if k == attempt_key},
        }
        episodes = evaluation.read_episodes([cell["episode_log_path"]])
        evaluation.validate_results(episodes, subset, require_primary_checkpoint=True)
        if len(episodes) != 500:
            raise ValueError("Completed suite must contain all 500 prescribed episodes")
        first = episodes[0]
        check_identity(
            {
                "code_commit": first["_provenance"]["source"]["commit"],
                "container_image_id": first["server_metadata"]["runtime"]["container_image"],
            }
        )
        evaluation_cells.add(key)
        receipts.append(
            {
                **record,
                "episode_log_path": cell["episode_log_path"],
                "episode_log_sha256": sha256_file(Path(cell["episode_log_path"])),
            }
        )
    return training, evaluation_cells, receipts


def choose_study(profile: dict, hourly_rate_usd: float, budget_usd: float) -> dict:
    """Choose full study, then the fixed seed-42 pilot, without reducing exposure.

    Input durations must bound each suite/group/K on the chosen hardware. A
    successful reference rollout alone is not a duration bound for a weak policy.
    Counts are derived here, rather than accepted from a hand-edited matrix.
    """
    if set(profile.get("seconds_per_optimizer_update", {})) != {"B0", "B1", "P"}:
        raise ValueError("Study selection requires complete optimizer profiles for B0, B1, and P")
    durations = profile.get("evaluation_seconds_per_episode")
    if not isinstance(durations, dict) or set(durations) != {"B0", "B1", "P", "reference"}:
        raise ValueError("Provide evaluation durations for all three arms and the released reference")
    if not profile.get("evaluation_measurement_source"):
        raise ValueError("Retain and identify the evaluation timing measurements")
    checkpoint_times = profile.get("checkpoint_seconds_per_save", {})
    checkpoint_intervals = profile.get("checkpoint_interval_updates", {})
    if set(checkpoint_times) != {"B0", "B1", "P"} or set(checkpoint_intervals) != {"B0", "B1", "P"}:
        raise ValueError("Provide measured checkpoint_seconds_per_save and checkpoint_interval_updates for each arm")
    for group in ("B0", "B1", "P"):
        positive_number(checkpoint_times[group], f"checkpoint_seconds_per_save.{group}")
        interval = checkpoint_intervals[group]
        if type(interval) is not int or not 1 <= interval <= 5000:
            raise ValueError("checkpoint_interval_updates must be integers in 1..5000")
    completed_training, completed_evaluation, receipts = _study_progress(profile)
    suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
    studies = {}
    for study, seeds in (("full", (17, 42, 93)), ("pilot", (42,))):
        seed_count = len(seeds)
        cells = []
        remaining_eval_seconds = 0.0
        completed_episodes = 0
        for group in ("B0", "B1", "P", "reference"):
            if set(durations[group]) != {"1", "10"}:
                raise ValueError("Evaluation durations must cover K=1 and K=10")
            for steps in ("1", "10"):
                if set(durations[group][steps]) != set(suites):
                    raise ValueError("Evaluation durations must cover every native LIBERO suite")
                for suite in suites:
                    cells.append(
                        {
                            "group": group,
                            "num_steps": int(steps),
                            "suite": suite,
                            "episodes": 500 * (1 if group == "reference" else seed_count),
                            "seconds_per_episode": durations[group][steps][suite],
                        }
                    )
                    for seed in (None,) if group == "reference" else seeds:
                        if (group, seed, int(steps), suite) in completed_evaluation:
                            completed_episodes += 500
                        else:
                            remaining_eval_seconds += 500 * positive_number(
                                durations[group][steps][suite], "seconds_per_episode"
                            )
        scenario = {
            **profile,
            "training_seeds": seed_count,
            "evaluation_cells": cells,
            "evaluation_episodes": sum(cell["episodes"] for cell in cells),
        }
        forecast = estimate_budget(scenario, hourly_rate_usd, budget_usd)
        remaining_train_seconds = checkpoint_seconds = 0.0
        completed_updates = remaining_saves = 0
        for group in ("B0", "B1", "P"):
            for seed in seeds:
                done = completed_training.get((group, seed), 0)
                completed_updates += done
                remaining_train_seconds += (30_000 - done) * profile["seconds_per_optimizer_update"][group]
                interval = checkpoint_intervals[group]
                saves = 30_000 // interval - done // interval + int(done < 30_000 and 30_000 % interval != 0)
                remaining_saves += saves
                checkpoint_seconds += saves * checkpoint_times[group]
        remaining_compute = (
            hourly_rate_usd * (remaining_train_seconds + remaining_eval_seconds + checkpoint_seconds) / 3600
        )
        total = forecast["fixed_cost_usd"] + 1.2 * remaining_compute
        studies[study] = {
            **forecast,
            "planned_training_gpu_hours": forecast["training_gpu_hours"],
            "planned_evaluation_gpu_hours": forecast["evaluation_gpu_hours"],
            "training_gpu_hours": remaining_train_seconds / 3600,
            "evaluation_gpu_hours": remaining_eval_seconds / 3600,
            "checkpoint_gpu_hours": checkpoint_seconds / 3600,
            "planned_optimizer_updates": seed_count * 3 * 30_000,
            "completed_optimizer_updates": completed_updates,
            "remaining_optimizer_updates": seed_count * 3 * 30_000 - completed_updates,
            "planned_evaluation_episodes": scenario["evaluation_episodes"],
            "completed_evaluation_episodes": completed_episodes,
            "remaining_evaluation_episodes": scenario["evaluation_episodes"] - completed_episodes,
            "remaining_checkpoint_saves": remaining_saves,
            "gpu_cost_before_headroom_usd": remaining_compute,
            "remaining_compute_cost_with_headroom_usd": 1.2 * remaining_compute,
            "projected_total_usd": total,
            "within_budget": total <= budget_usd,
        }
    selected = next((name for name in ("full", "pilot") if studies[name]["within_budget"]), None)
    return {
        "schema_version": 1,
        "selected_study": selected,
        "budget_usd": budget_usd,
        "hourly_rate_usd": hourly_rate_usd,
        "studies": studies,
        "training_seeds": [17, 42, 93] if selected == "full" else ([42] if selected == "pilot" else []),
        "optimizer_updates": 30000,
        "effective_batch_size": 256,
        "remaining_compute_cost_with_headroom_usd": (
            studies[selected]["remaining_compute_cost_with_headroom_usd"] if selected else None
        ),
        "measurement_source": profile["measurement_source"],
        "evaluation_measurement_source": profile["evaluation_measurement_source"],
        "checkpoint_interval_updates": dict(checkpoint_intervals),
        "experiment_identity": profile.get("experiment_identity", {}),
        "validated_progress": receipts,
        "note": "Remaining work only; fixed cost includes all spending so far and noncompute reserves. No shortened-run fallback.",
    }


def inspect_stack(repo: Path) -> dict:
    packages = {}
    for name, expected in REQUIRED_PACKAGES.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {
            "installed": version,
            "expected": expected,
            "matches": version is not None and version.split("+")[0] == expected,
        }

    patch = {"matches": False, "reason": "Transformers package unavailable"}
    if packages["transformers"]["installed"]:
        spec = importlib.util.find_spec("transformers")
        if spec is not None and spec.origin:
            installed_root = Path(spec.origin).parent
            source_root = repo / "src/openpi/models_pytorch/transformers_replace"
            source_files = sorted(source_root.rglob("*.py"))
            mismatched = []
            for source in source_files:
                relative = source.relative_to(source_root)
                target = installed_root / relative
                if not target.is_file() or sha256_file(source) != sha256_file(target):
                    mismatched.append(str(relative))
            patch = {
                "matches": bool(source_files) and not mismatched,
                "checked_files": len(source_files),
                "mismatched": mismatched,
            }

    cuda = {"available": False, "full_model_double_backward": "not tested"}
    try:
        import torch

        cuda["available"] = torch.cuda.is_available()
        cuda["torch_cuda_runtime"] = torch.version.cuda
        cuda["cudnn_version"] = torch.backends.cudnn.version()
        if cuda["available"]:
            cuda["devices"] = [
                {
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except (ImportError, OSError, RuntimeError) as exc:
        cuda["error"] = str(exc)

    return {
        "python": sys.version.split()[0],
        "python_matches": sys.version_info[:2] == (3, 11),
        "platform": platform.platform(),
        "canonical_platform": sys.platform == "linux",
        "packages": packages,
        "transformers_patch": patch,
        "cuda": cuda,
    }


def inspect_assets(checkpoint: Path | None, dataset: Path | None, norm_stats: Path | None, kind: str) -> dict:
    result = {"checkpoint": {"ready": False}, "dataset": {"ready": False}, "normalization": {"ready": False}}
    if checkpoint is not None:
        record = {"path": str(checkpoint.resolve()), "kind": kind, "ready": False}
        weights = checkpoint / "model.safetensors"
        record["weights_present"] = weights.is_file() and weights.stat().st_size > 0
        if weights.is_file():
            record["weights_bytes"] = weights.stat().st_size
        metadata_path = checkpoint / "potential_metadata.json"
        record["architecture_metadata_required"] = kind == "research"
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            record["metadata"] = metadata
            record["metadata_sha256"] = sha256_file(metadata_path)
            record["metadata_valid"] = (
                metadata.get("schema_version") == 1
                and metadata.get("velocity_mode") in ("direct", "potential_dot")
                and metadata.get("action_horizon") == 10
                and metadata.get("action_dim") == 32
            )
        else:
            record["metadata_valid"] = kind == "base"
            record["metadata_status"] = (
                "missing" if kind == "research" else "upstream base input, not a research checkpoint"
            )
        record["ready"] = record["weights_present"] and record["metadata_valid"]
        result["checkpoint"] = record
    if dataset is not None:
        info_path = dataset / "meta/info.json"
        record = {"path": str(dataset.resolve()), "ready": False, "info_present": info_path.is_file()}
        if info_path.is_file():
            info = read_json(info_path)
            record["metadata"] = {
                key: info.get(key)
                for key in ("total_episodes", "total_frames", "total_tasks", "codebase_version", "splits")
            }
            record["info_sha256"] = sha256_file(info_path)
            record["parquet_present"] = next(dataset.glob("data/**/*.parquet"), None) is not None
            record["reference_counts_match"] = (
                info.get("total_tasks") == 40
                and info.get("total_episodes") == 1693
                and info.get("total_frames") == 273465
            )
            record["ready"] = record["parquet_present"] and record["reference_counts_match"]
            record["content_integrity"] = (
                "metadata and presence checked; full dataset hashes must be recorded separately"
            )
        result["dataset"] = record
    if norm_stats is not None:
        record = {"path": str(norm_stats.resolve()), "ready": False}
        if norm_stats.is_file():
            data = read_json(norm_stats)
            stats = data.get("norm_stats", data)
            actions = stats.get("actions", {})
            record["ready"] = isinstance(actions, dict) and "q01" in actions and "q99" in actions
            record["sha256"] = sha256_file(norm_stats)
        result["normalization"] = record
    return result


def git_report(repo: Path) -> dict:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=repo, capture_output=True, check=False
        )
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True)
        return {
            "head": head,
            "expected_upstream": BASE_COMMIT,
            "upstream_is_ancestor": ancestry.returncode == 0,
            "dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"expected_upstream": BASE_COMMIT, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-kind", choices=("base", "research"), default="research")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--norm-stats", type=Path)
    parser.add_argument("--profile", type=Path, help="JSON containing real full-model GPU measurements")
    parser.add_argument(
        "--select-study", action="store_true", help="Choose full/pilot from per-arm and per-suite timings"
    )
    parser.add_argument("--hourly-rate-usd", type=float)
    parser.add_argument("--budget-usd", type=float, default=2000.0)
    parser.add_argument("--report", type=Path, help="Optionally save the exact JSON report")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--require-profile", action="store_true")
    parser.add_argument("--strict-stack", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    stack = inspect_stack(repo)
    errors = []
    try:
        assets = inspect_assets(args.checkpoint, args.dataset, args.norm_stats, args.checkpoint_kind)
    except (OSError, ValueError, TypeError) as exc:
        assets = {"error": str(exc)}
        errors.append("Asset inspection failed")
    budget = {"status": "not measured"}
    if args.profile is not None:
        try:
            budget = (choose_study if args.select_study else estimate_budget)(
                read_json(args.profile), args.hourly_rate_usd, args.budget_usd
            )
        except (OSError, ValueError, TypeError) as exc:
            budget = {"status": "invalid measurement record", "error": str(exc)}
            errors.append("Budget profile failed validation")
    stack_ready = (
        stack["python_matches"]
        and stack["canonical_platform"]
        and all(item["matches"] for item in stack["packages"].values())
        and stack["transformers_patch"]["matches"]
    )
    assets_ready = all(assets.get(key, {}).get("ready", False) for key in ("checkpoint", "dataset", "normalization"))
    if args.strict_stack and not stack_ready:
        errors.append("Canonical Linux/Python/package/patch requirements are not satisfied")
    if args.require_cuda and not stack["cuda"]["available"]:
        errors.append("CUDA is unavailable")
    if args.require_assets and not assets_ready:
        errors.append("Required checkpoint, dataset, or normalization assets are unavailable or invalid")
    if args.require_profile and not (budget.get("within_budget", False) or budget.get("selected_study")):
        errors.append("No qualifying measured full-model GPU profile within the budget")
    report = {
        "recorded_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "source": git_report(repo),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
        "stack": stack,
        "assets": assets,
        "budget": budget,
        "ready_for_full_model_gpu_smoke_test": stack_ready and assets_ready and stack["cuda"]["available"],
        "validated_training_run": False,
        "limitations": [
            "No complete pi05 forward/backward or rollout is executed by this audit.",
            "CPU and synthetic tests do not establish GPU memory, throughput, or benchmark performance.",
        ],
        "errors": errors,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
