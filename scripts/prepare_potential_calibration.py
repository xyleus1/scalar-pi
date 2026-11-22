"""Create or verify the exact calibration specification after all GPU profiles pass."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_potential_evaluation as calibration
import select_potential_batch as selection

CONFIGS = {"B0": "pi05_libero_b0", "B1": "pi05_libero_b1", "P": "pi05_libero_potential"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_profiles(data):
    data = Path(data)
    reports = data / "experiments/reports"
    args = SimpleNamespace(
        weights=data / "weights/pi05_base_torch",
        data_manifest=data / "libero_manifest.json",
        qualification_report=reports / "gpu_qualification.json",
        recovery_report=reports / "recovery.json",
        verify_report=data / "experiments/batch_selection.json",
    )
    selection.verify_report(args)
    result = {}
    for group, config in CONFIGS.items():
        args.config = config
        args.verify_profile = reports / "profiles" / f"{config}.json"
        result[group] = selection.verify_profile(args)
    return result


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def host_path(path, root):
    path = Path(path)
    return root / "data" / path.relative_to("/data") if path.is_relative_to("/data") else path


def verify_aggregate(spec, path):
    repository = Path(__file__).resolve().parents[1]
    source = calibration.ev.source_identity(repository)
    model_image = os.environ.get("SCALAR_PI_MODEL_IMAGE")
    simulator_image = os.environ.get("SCALAR_PI_SIMULATOR_IMAGE")
    if source["tracked_changes"] or not model_image or not simulator_image:
        raise ValueError("Calibration reuse requires clean source and both container identities")
    actual = calibration.compose(
        spec,
        calibration.commands(spec),
        source=source,
        model_image=model_image,
        simulator_image=simulator_image,
        corpus_hash=calibration.ev.file_digest(spec["corpus"]),
        checkpoint_hashes={
            item["group"]: calibration.checkpoint_hash(item["checkpoint_dir"]) for item in spec["policies"]
        },
    )
    if read_json(path) != actual:
        raise ValueError("Published calibration aggregate differs from current component evidence")


def prepare(root, *, attempt, smoke_manifest=None):
    root = Path(root)
    data, reports = root / "data", root / "data/experiments/reports"
    profiles = verified_profiles(data)
    target = reports / "evaluation_calibration_spec.json"
    existing = read_json(target) if target.exists() else None
    if smoke_manifest is None and existing:
        smoke_manifest = Path(existing["smoke_manifest"])
    if smoke_manifest is None:
        complete = []
        for path in sorted((reports / "smoke").glob("*/matrix.json")):
            try:
                calibration.smoke_overhead(path)
            except (OSError, ValueError, KeyError):
                continue
            complete.append(path)
        if len(complete) != 1:
            raise ValueError("Exactly one completed smoke is required; use --smoke-manifest to identify another")
        smoke_manifest = complete[0]
    calibration.smoke_overhead(smoke_manifest)
    template = Path(__file__).resolve().parents[1] / "configs/potential/evaluation_calibration.example.json"
    spec = read_json(template)
    runner = str(Path(__file__).resolve().parent / "docker/run_potential.sh")
    spec["policy_command_prefix"] = ["bash", runner, "model", "--execute", "python", "scripts/benchmark_potential.py"]
    spec["simulator_command_prefix"] = ["bash", runner, "simulator", "--execute"]
    spec["output_dir"] = existing["output_dir"] if existing else str(data / "experiments/calibration" / attempt)
    spec["corpus"] = str(data / "policy_corpus.npz")
    spec["smoke_manifest"] = str(smoke_manifest)
    for policy in spec["policies"]:
        group = policy["group"]
        if group == "reference":
            policy["command_prefix"] = [
                "bash",
                runner,
                "model-jax",
                "--execute",
                "python",
                "scripts/benchmark_potential.py",
            ]
            policy["checkpoint_dir"] = str(data / "openpi-cache/openpi-assets/checkpoints/pi05_libero")
        else:
            profile = profiles[group]
            run = host_path(Path(profile["measurement_source"]).parent, root)
            policy["checkpoint_dir"] = str(run / str(profile["completed_updates"]))
            if policy["config"] != CONFIGS[group]:
                raise ValueError("Calibration template changed its research arms")
    if existing:
        if existing != spec:
            raise ValueError("Existing calibration specification differs; preserve and investigate its evidence")
    else:
        publish(target, spec)
    aggregate = reports / "evaluation_calibration.json"
    if aggregate.exists():
        verify_aggregate(spec, aggregate)
    print(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("SCALAR_PI_ROOT", str(Path(__file__).resolve().parents[1] / ".research"))),
    )
    parser.add_argument("--smoke-manifest", type=Path)
    parser.add_argument("--attempt", default=os.environ.get("SCALAR_PI_ATTEMPT", "calibration"))
    args = parser.parse_args(argv)
    prepare(args.root, attempt=args.attempt, smoke_manifest=args.smoke_manifest)


if __name__ == "__main__":
    main()
