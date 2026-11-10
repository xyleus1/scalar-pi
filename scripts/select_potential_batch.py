"""Select a common microbatch using complete potential-model optimizer updates.

Preview is the default. --execute runs short CUDA diagnostics, not cost profiles.
Each candidate uses a fresh trainer process and retains logs, without saving weights.
The selected batch must separately pass publication and recovery qualification.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

CONFIG = "pi05_libero_potential"
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def input_hashes(args):
    return {
        "weights": digest(args.weights / "model.safetensors"),
        "conversion_receipt": digest(args.weights / "config.json"),
        "data_manifest": digest(args.data_manifest),
        "qualification": digest(args.qualification_report),
    }


def verify_report(args, *, repository=None):
    """Check existing fit evidence against current source/assets, without loading a model."""
    report = json.loads(args.verify_report.read_text())
    if (
        report.get("schema_version") != 1
        or report.get("verified") is not True
        or report.get("kind") != "batch_selection"
        or report.get("effective_batch_size") != 256
        or type(report.get("microbatch_size")) is not int
        or report.get("microbatch_size") not in BATCHES
        or type(report.get("gradient_checkpointing")) is not bool
        or report.get("matmul_precision") != "highest"
    ):
        raise ValueError("Invalid batch-selection report")
    image = os.environ.get("SCALAR_PI_IMAGE_ID")
    if not image or report.get("container_image_id") != image:
        raise ValueError("Batch selection container differs from the current environment")
    if report.get("inputs_sha256") != input_hashes(args):
        raise ValueError("Batch selection inputs differ from the current staged inputs")
    repository = repository or Path(__file__).resolve().parents[1]
    code = report.get("code", {})
    for command, expected in ((["rev-parse", "HEAD"], code.get("commit")), (["status", "--porcelain"], "")):
        actual = subprocess.run(["git", *command], cwd=repository, capture_output=True, text=True, check=True)
        if actual.stdout.strip() != expected:
            raise ValueError("Batch selection requires the same clean source commit")
    sources = code.get("source_files_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("Batch selection is missing actual source hashes")
    for name, expected in sources.items():
        path = (repository / name).resolve()
        if not path.is_relative_to(repository.resolve()) or digest(path) != expected:
            raise ValueError(f"Batch selection source differs: {name}")
    if hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest() != code.get("source_snapshot_sha256"):
        raise ValueError("Batch selection source digest is inconsistent")
    for config in ("pi05_libero_b0", "pi05_libero_b1", CONFIG):
        norms = repository / "assets" / config / "physical-intelligence/libero/norm_stats.json"
        if digest(norms) != report.get("normalization_sha256"):
            raise ValueError(f"Batch selection normalization differs for {config}")
    return report


def command(args, batch, checkpointing):
    name = f"batch_{batch}_{'checkpointed' if checkpointing else 'ordinary'}"
    argv = [
        sys.executable,
        str(Path(__file__).with_name("train_potential.py")),
        CONFIG,
        "--exp-name",
        name,
        "--weights",
        str(args.weights),
        "--data-manifest",
        str(args.data_manifest),
        "--output-root",
        str(args.output_root),
        "--microbatch-size",
        str(batch),
        "--fit-probe",
        "--qualification-report",
        str(args.qualification_report),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--hourly-rate",
        str(args.hourly_rate),
        "--matmul-precision",
        "highest",
        "--deterministic",
        "--gradient-checkpointing" if checkpointing else "--no-gradient-checkpointing",
    ]
    if args.deadline_unix is not None:
        argv.extend(("--deadline-unix", str(args.deadline_unix)))
    return argv, args.output_root / CONFIG / name


def _qualified(args):
    from openpi.research.qualification import verify_qualification

    return verify_qualification(args.qualification_report, checkpoint=args.weights)


def verify_profile(args):
    """Reuse one completed arm only while its measurements and inputs remain intact."""
    from openpi.research.qualification import verify_recovery

    qualification = _qualified(args)
    profile = json.loads(args.verify_profile.read_text())
    if (
        profile.get("mode") != "profile"
        or profile.get("measured_gpu") is not True
        or profile.get("valid_training_cost_profile") is not True
        or profile.get("completed_updates") != 110
        or profile.get("warmup_optimizer_updates") != 10
        or profile.get("timed_optimizer_updates") != 100
        or profile.get("effective_batch_size") != 256
        or profile.get("identity") != qualification["identity"]
    ):
        raise ValueError("Reuse requires a complete matching 10-warmup/100-update GPU profile")
    for field in ("measurement", "run_manifest", "checkpoint_measurements"):
        path = Path(profile["measurement_source" if field == "measurement" else field])
        if digest(path) != profile.get(f"{field}_sha256"):
            raise ValueError(f"Profile retained evidence differs: {field}")
    manifest = json.loads(Path(profile["run_manifest"]).read_text())
    protocol = manifest["protocol"]
    identity = qualification["identity"]
    expected = {
        "source_weights_sha256": identity["base_checkpoint_sha256"],
        "base_conversion_receipt_sha256": identity["base_conversion_sha256"],
        "data_manifest_sha256": digest(args.data_manifest),
        "normalization_hash": qualification["normalization_sha256"],
        "qualification_sha256": digest(args.qualification_report),
        "recovery_sha256": digest(args.recovery_report),
        "container_image_id": identity["container_image_id"],
        "mode": "profile",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Profile inputs differ from current staged inputs")
    if (
        protocol.get("config") != args.config
        or protocol.get("matmul_precision") != "highest"
        or protocol.get("deterministic") is not True
        or protocol.get("code_commit") != identity["code_commit"]
        or profile.get("qualification_sha256") != expected["qualification_sha256"]
        or profile.get("recovery_sha256") != expected["recovery_sha256"]
    ):
        raise ValueError("Profile arithmetic/configuration differs from the qualified protocol")
    verify_recovery(
        args.recovery_report,
        identity=identity,
        microbatch_size=protocol["microbatch_size"],
        gradient_checkpointing=protocol["gradient_checkpointing"],
        data_manifest=args.data_manifest,
        normalization_hash=qualification["normalization_sha256"],
    )
    metrics = [json.loads(line) for line in Path(profile["measurement_source"]).read_text().splitlines()]
    if [row["optimizer_updates"] for row in metrics] != list(range(1, 111)):
        raise ValueError("Profile must retain all 110 complete optimizer-update measurements")
    timings = [row["seconds_per_update"] for row in metrics]
    if any(not isinstance(value, int | float) or not 0 < value < float("inf") for value in timings):
        raise ValueError("Profile contains invalid update measurements")
    if not math.isclose(sum(timings[10:]) / 100, profile["seconds_per_optimizer_update"][args.config]):
        raise ValueError("Profile summary differs from retained measurements")
    publications = [json.loads(line) for line in Path(profile["checkpoint_measurements"]).read_text().splitlines()]
    if not publications or publications[-1]["optimizer_updates"] != 110:
        raise ValueError("Profile is missing final publication measurements")
    if [row["seconds"] for row in publications] != profile.get("checkpoint_seconds"):
        raise ValueError("Profile publication summary differs from retained measurements")
    checkpoint = Path(publications[-1]["checkpoint"])
    metadata = json.loads((checkpoint / "potential_metadata.json").read_text())
    if metadata.get("run_mode") != "profile" or metadata.get("optimizer_updates") != 110:
        raise ValueError("Profile checkpoint is not the final diagnostic boundary")
    for filename, key in (("model.safetensors", "model_sha256"), ("raw_model.safetensors", "raw_model_sha256")):
        if digest(checkpoint / filename) != metadata.get(key):
            raise ValueError("Profile published checkpoint bytes differ")
    return profile


def select(args):
    if args.output_root.exists() or args.output.exists():
        raise FileExistsError("Use a fresh selection output directory and report path")
    image_id = os.environ.get("SCALAR_PI_IMAGE_ID")
    if not image_id:
        raise ValueError("Set SCALAR_PI_IMAGE_ID to the actual model container identity")
    _qualified(args)
    inputs = input_hashes(args)
    args.output_root.mkdir(parents=True)
    attempts = []
    selected = None
    for checkpointing in (False, True):
        for batch in BATCHES:
            argv, directory = command(args, batch, checkpointing)
            log = args.output_root / f"{directory.name}.log"
            attempt = {"microbatch_size": batch, "gradient_checkpointing": checkpointing, "log": str(log)}
            attempts.append(attempt)
            with log.open("wb") as stream:
                options = {}
                if args.deadline_unix is not None:
                    remaining = args.deadline_unix - time.time() - 600
                    if remaining <= 0:
                        raise TimeoutError("Candidate would consume the shutdown reserve")
                    options["timeout"] = remaining
                process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=False, **options)
            if process.returncode:
                error = log.read_text(encoding="utf-8", errors="replace")
                if re.search(r"(?:torch\.(?:cuda\.)?OutOfMemoryError|RuntimeError): CUDA out of memory", error):
                    attempt["status"] = "cuda_out_of_memory"
                    break
                raise RuntimeError(f"Candidate failed with exit {process.returncode}; inspect {log}")
            profile_path = directory / "profile.json"
            profile = json.loads(profile_path.read_text())
            if (
                profile.get("measured_gpu") is not True
                or profile.get("mode") != "fit"
                or profile.get("fit_probe_passed") is not True
                or profile.get("valid_training_cost_profile") is not False
                or profile.get("completed_updates") != 2
                or profile.get("timed_optimizer_updates") != 1
                or profile.get("effective_batch_size") != 256
                or profile.get("checkpoint_seconds") != []
            ):
                raise ValueError(
                    f"Candidate did not complete two real optimizer/EMA updates without saving: {profile_path}"
                )
            total, reserved = profile.get("total_memory_bytes"), profile.get("peak_reserved_bytes")
            if type(total) is not int or type(reserved) is not int or total <= 0 or reserved <= 0:
                raise ValueError(f"Missing GPU memory measurements: {profile_path}")
            attempt.update(
                profile=str(profile_path),
                profile_sha256=digest(profile_path),
                reserved_fraction=reserved / total,
                status="fit" if reserved < 0.9 * total else "headroom_limit",
            )
            if attempt["status"] != "fit":
                break
            selected = attempt
        if selected is not None:
            break  # Checkpointing is a fallback only when even ordinary batch one cannot fit.
    if selected is None:
        raise RuntimeError(f"No potential-model microbatch fits with 10% memory headroom; logs: {args.output_root}")
    directory = Path(selected["profile"]).parent
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("container_image_id") != image_id:
        raise ValueError("Selected candidate container differs from the current environment")
    if (
        manifest.get("source_weights_sha256") != inputs["weights"]
        or manifest.get("base_conversion_receipt_sha256") != inputs["conversion_receipt"]
        or manifest.get("data_manifest_sha256") != inputs["data_manifest"]
    ):
        raise ValueError("Selected candidate inputs differ from the requested verified inputs")
    result = {
        "schema_version": 1,
        "kind": "batch_selection",
        "verified": True,
        "valid_training_cost_profile": False,
        "requires_recovery_qualification": True,
        "microbatch_size": selected["microbatch_size"],
        "gradient_checkpointing": selected["gradient_checkpointing"],
        "effective_batch_size": 256,
        "matmul_precision": "highest",
        "seed": args.seed,
        "hourly_rate_usd": args.hourly_rate,
        "container_image_id": image_id,
        "code": manifest["code"],
        "inputs_sha256": inputs,
        "normalization_sha256": manifest["normalization_hash"],
        "selected_profile": selected["profile"],
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--weights", dest="weights", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--recovery-report", type=Path)
    parser.add_argument("--verify-profile", type=Path, help="Verify one completed cost profile and retained evidence")
    parser.add_argument("--config", choices=("pi05_libero_b0", "pi05_libero_b1", CONFIG))
    parser.add_argument("--deadline-unix", type=float, default=os.environ.get("SCALAR_PI_DEADLINE_UNIX") or None)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--report", "--output", dest="output", type=Path)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument(
        "--verify-report", type=Path, help="Read-only check of a prior selection against current inputs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_profile:
        if args.verify_report or args.execute or args.recovery_report is None or args.config is None:
            parser.error("--verify-profile requires --config/--recovery-report and cannot execute or verify selection")
        print(json.dumps(verify_profile(args), indent=2))
        return
    if args.verify_report:
        if args.execute:
            parser.error("--verify-report does not execute profiling")
        print(json.dumps(verify_report(args), indent=2))
        return
    if args.output_root is None or args.output is None or args.hourly_rate is None:
        parser.error("Selection requires --output-root, --report and --hourly-rate")
    if not math.isfinite(args.hourly_rate) or args.hourly_rate <= 0 or args.num_workers < 0:
        parser.error("Require a positive hourly rate and nonnegative worker count")
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "batch_candidates": BATCHES,
                    "first_command": command(args, 1, checkpointing=False)[0],
                    "checkpointing": "Only if ordinary batch one cannot fit with 10% headroom",
                },
                indent=2,
            )
        )
        return
    print(json.dumps(select(args), indent=2))


if __name__ == "__main__":
    main()
