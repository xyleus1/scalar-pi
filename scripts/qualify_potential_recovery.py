"""Qualify the selected full-size P batch using exact CUDA save/resume and serving.

Dry run by default. --execute runs three isolated trainer processes: two updates
uninterrupted, one update, then resume to two. Diagnostics remain separate from
research runs. No cloud resource is created and no assets are downloaded.
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CONFIG = "pi05_libero_potential"


def trainer_command(args, selection, *, name, updates, resume=None):
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_potential.py")),
        CONFIG,
        "--exp-name",
        name,
        "--output-root",
        str(args.output_root),
        "--data-manifest",
        str(args.data_manifest),
        "--qualification-report",
        str(args.qualification_report),
        "--microbatch-size",
        str(selection["microbatch_size"]),
        "--seed",
        "42",
        "--num-workers",
        str(args.num_workers),
        "--recovery-probe",
        "--updates",
        str(updates),
        "--save-every",
        str(updates),
        "--keep-last-checkpoints",
        "2",
        "--gradient-checkpointing" if selection["gradient_checkpointing"] else "--no-gradient-checkpointing",
    ]
    command.extend(("--resume", str(resume)) if resume else ("--weights", str(args.checkpoint)))
    if args.deadline_unix is not None:
        command.extend(("--deadline-unix", str(args.deadline_unix)))
    return command


def run_stage(command, log, deadline):
    remaining = None if deadline is None else deadline - time.time() - 600
    if remaining is not None and remaining <= 0:
        raise TimeoutError("Recovery stage would consume the shutdown reserve")
    with log.open("wb") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=remaining)


def assert_equal(left, right, name="state"):
    """Exact nested comparison including optimizer, EMA, RNG and sample cursor."""
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(left, right):
            raise ValueError(f"Recovery differs: {name}")
    elif isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
            raise ValueError(f"Recovery differs: {name}")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError(f"Recovery keys differ: {name}")
        for key in left:
            assert_equal(left[key], right[key], f"{name}.{key}")
    elif isinstance(left, tuple | list):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError(f"Recovery sequence differs: {name}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            assert_equal(a, b, f"{name}[{index}]")
    elif type(left) is not type(right) or left != right:
        raise ValueError(f"Recovery differs: {name}")


def compare_checkpoints(continuous, resumed):
    from safetensors import safe_open
    import torch

    for filename in ("raw_model.safetensors", "model.safetensors"):
        with (
            safe_open(continuous / filename, framework="pt", device="cpu") as left,
            safe_open(resumed / filename, framework="pt", device="cpu") as right,
        ):
            if set(left.keys()) != set(right.keys()):
                raise ValueError(f"Recovery tensor keys differ: {filename}")
            for key in left.keys():  # noqa: SIM118 -- safe_open is not a mapping.
                assert_equal(left.get_tensor(key), right.get_tensor(key), f"{filename}.{key}")
    left = torch.load(continuous / "training_state.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed / "training_state.pt", map_location="cpu", weights_only=False)
    # Wall-clock training time must differ; all actual training state must agree.
    for state in (left, right):
        state.pop("training_seconds")
    assert_equal(left, right)
    if left["optimizer_updates"] != 2:
        raise ValueError("Recovery must compare the same two-update boundary")
    return {"raw_weights": True, "ema_weights": True, "optimizer_ema_rng_cursor": True}


def verify_inputs(args):
    from openpi.research.evaluation import file_digest
    from openpi.research.qualification import verify_qualification

    spec = importlib.util.spec_from_file_location(
        "batch_selection", Path(__file__).with_name("select_potential_batch.py")
    )
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    qualification = verify_qualification(args.qualification_report, checkpoint=args.checkpoint)
    selection = selector.verify_report(
        argparse.Namespace(
            verify_report=args.selection,
            weights=args.checkpoint,
            data_manifest=args.data_manifest,
            qualification_report=args.qualification_report,
        )
    )
    if file_digest(args.corpus) != qualification["corpus_sha256"]:
        raise ValueError("Recovery corpus differs from numerical qualification")
    return qualification, selection


def verify_report(args):
    from openpi.research.evaluation import file_digest
    from openpi.research.qualification import verify_recovery

    qualification, selection = verify_inputs(args)
    report = verify_recovery(
        args.verify_report,
        identity=qualification["identity"],
        microbatch_size=selection["microbatch_size"],
        gradient_checkpointing=selection["gradient_checkpointing"],
        data_manifest=args.data_manifest,
        normalization_hash=qualification["normalization_sha256"],
    )
    for field, path in (
        ("selection", args.selection),
        ("qualification", args.qualification_report),
        ("corpus", args.corpus),
    ):
        if report.get(f"{field}_sha256") != file_digest(path):
            raise ValueError(f"Recovery {field} changed")
    serving = report["actual_policy_serving"]
    if file_digest(serving["report"]) != serving["sha256"]:
        raise ValueError("Recovery serving evidence changed")
    if file_digest(Path(report["checkpoint"]) / "model.safetensors") != report["checkpoint_sha256"]:
        raise ValueError("Recovery published checkpoint changed")
    return report


def qualify(args):
    from openpi.research.evaluation import file_digest
    from openpi.research.qualification import verify_qualification

    if args.output_root.exists() or args.report.exists():
        raise FileExistsError("Recovery evidence requires fresh output/report paths")
    qualification, selection = verify_inputs(args)
    report = {
        "schema_version": 1,
        "purpose": "full_size_cuda_recovery",
        "passed": False,
        "measured_gpu": True,
        "identity": qualification["identity"],
        "microbatch_size": selection["microbatch_size"],
        "gradient_checkpointing": selection["gradient_checkpointing"],
        "data_manifest_sha256": file_digest(args.data_manifest),
        "normalization_sha256": qualification["normalization_sha256"],
        "qualification_sha256": file_digest(args.qualification_report),
        "selection_sha256": file_digest(args.selection),
        "corpus_sha256": file_digest(args.corpus),
        "valid_training_cost_profile": False,
        "valid_benchmark_evaluation": False,
    }
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    continuous = args.output_root / CONFIG / "continuous" / "2"
    split = args.output_root / CONFIG / "resumed"
    try:
        if selection["gradient_checkpointing"]:
            math_report = args.output_root / "checkpointed_math.json"
            run_stage(
                [
                    sys.executable,
                    str(Path(__file__).with_name("qualify_potential.py")),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--math-only",
                    "--gradient-checkpointing",
                    "--report",
                    str(math_report),
                ],
                args.output_root / "checkpointed_math.log",
                args.deadline_unix,
            )
            math_result = verify_qualification(math_report, checkpoint=args.checkpoint, require_serving=False)
            if math_result.get("gradient_checkpointing") is not True:
                raise ValueError("Selected checkpointing mode was not qualified")
            report["checkpointed_math_sha256"] = file_digest(math_report)
        stages = (
            trainer_command(args, selection, name="continuous", updates=2),
            trainer_command(args, selection, name="resumed", updates=1),
            trainer_command(args, selection, name="resumed", updates=2, resume=split / "1"),
        )
        for index, command in enumerate(stages):
            run_stage(command, args.output_root / f"trainer_{index + 1}.log", args.deadline_unix)
        report["exact_resume"] = compare_checkpoints(continuous, split / "2")
        serving_report = args.output_root / "serving.json"
        run_stage(
            [
                sys.executable,
                str(Path(__file__).with_name("qualify_potential.py")),
                "--checkpoint",
                str(split / "2"),
                "--checkpoint-kind",
                "research",
                "--serving-only",
                "--corpus",
                str(args.corpus),
                "--report",
                str(serving_report),
            ],
            args.output_root / "serving.log",
            args.deadline_unix,
        )
        serving = json.loads(serving_report.read_text())
        if not serving.get("correctness_checks_passed") or not serving.get("actual_policy_serving", {}).get("passed"):
            raise ValueError("Selected-batch P checkpoint did not pass actual serving")
        report["actual_policy_serving"] = {
            "passed": True,
            "report": str(serving_report),
            "sha256": file_digest(serving_report),
        }
        updates, checkpoints, profiles = [], [], []
        for directory in (continuous.parent, split):
            updates.extend(
                json.loads(line)["seconds_per_update"]
                for line in (directory / "metrics.jsonl").read_text().splitlines()
            )
            checkpoints.extend(
                json.loads(line) for line in (directory / "checkpoint_metrics.jsonl").read_text().splitlines()
            )
            profiles.append(json.loads((directory / "profile.json").read_text()))
        report.update(
            update_seconds=max(updates),
            checkpoint_seconds=max(row["seconds"] for row in checkpoints),
            checkpoint_measurements=checkpoints,
            peak_reserved_bytes=max(p["peak_reserved_bytes"] for p in profiles),
            startup=[p["startup"] for p in profiles],
            checkpoint=str(split / "2"),
            checkpoint_sha256=file_digest(split / "2" / "model.safetensors"),
            passed=True,
        )
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    report["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "data-manifest", "selection", "qualification-report", "corpus"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("output-root", "report", "verify-report"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--deadline-unix", type=float, default=os.environ.get("SCALAR_PI_DEADLINE_UNIX") or None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.num_workers < 0 or (args.deadline_unix is not None and not math.isfinite(args.deadline_unix)):
        parser.error("Require nonnegative workers and finite deadline")
    if args.verify_report:
        if args.execute:
            parser.error("--verify-report is read-only")
        print(json.dumps(verify_report(args), indent=2))
        return 0
    if args.output_root is None or args.report is None:
        parser.error("Qualification requires --output-root and --report")
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "stages": [
                        "P continuous 2 updates",
                        "P split 1 + resumed 1 update",
                        "exact state comparison",
                        "P actual serving K1/K10",
                    ],
                },
                indent=2,
            )
        )
        return 0
    report = qualify(args)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
