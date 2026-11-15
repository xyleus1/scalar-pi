"""Create or verify the exact calibration specification after all GPU profiles pass."""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import calibrate_potential_evaluation as calibration
from measured_budget import CONFIGS
from measured_budget import publish
from measured_budget import read_json
from measured_budget import verified_profiles


def host_path(path, root):
    path = Path(path)
    return root / "data" / path.relative_to("/data") if path.is_relative_to("/data") else path


def verify_aggregate(spec, path):
    repository = Path(__file__).resolve().parents[2]
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
    template = Path(__file__).resolve().parents[2] / "configs/potential/evaluation_calibration.example.json"
    spec = read_json(template)
    spec["output_dir"] = existing["output_dir"] if existing else str(data / "experiments/calibration" / attempt)
    spec["corpus"] = str(data / "policy_corpus.npz")
    spec["smoke_manifest"] = str(smoke_manifest)
    for policy in spec["policies"]:
        group = policy["group"]
        if group == "reference":
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
    parser.add_argument("--root", type=Path, default=Path("/lambda/nfs/scalar-pi"))
    parser.add_argument("--smoke-manifest", type=Path)
    args = parser.parse_args(argv)
    prepare(args.root, attempt=os.environ["SCALAR_PI_ATTEMPT"], smoke_manifest=args.smoke_manifest)


if __name__ == "__main__":
    main()
