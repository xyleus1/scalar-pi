"""Render the fixed full/pilot matrix and inventory persistent experiment artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def render_matrix(repo, root, study, *, diagnostic=False, attempt="default"):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", attempt):
        raise ValueError("Attempt must be a short filename component")
    filename = "libero_matrix.example.json" if study == "full" else "libero_matrix.pilot.json"
    matrix = json.loads((repo / "configs/potential" / filename).read_text())
    wrapper = str(repo / "scripts/lambda/run.sh")
    matrix["server_command_prefix"] = ["bash", wrapper, "server", "--execute"]
    matrix["client_command_prefix"] = ["bash", wrapper, "simulator", "--execute"]
    matrix["output_dir"] = str(root / "data/experiments/results" / study)
    for variant in matrix["variants"]:
        if variant.get("reference"):
            checkpoint = root / "data/openpi-cache/openpi-assets/checkpoints/pi05_libero"
        else:
            run_name = f"{variant['group'].lower()}_seed{variant['training_seed']}"
            checkpoint = root / "data/experiments/checkpoints" / variant["config"] / run_name / "30000"
        variant["checkpoint_dir"] = str(checkpoint)
    matrix["notes"] = "Rendered for Lambda; paths do not establish completed training. Validate actual checkpoints."
    if diagnostic:
        matrix.update(
            diagnostic=True,
            variants=[v for v in matrix["variants"] if v.get("reference")],
            training_seeds=[],
            num_steps=[10],
            num_tasks_per_suite=1,
            num_trials_per_task=1,
            output_dir=str(root / "data/experiments/results/diagnostic" / attempt),
            notes="Four released-policy compatibility episodes; not a benchmark result.",
        )
    return matrix


def archive(root):
    experiments = root / "data/experiments"
    entries = {}
    for path in sorted(experiments.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.json":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        entries[str(path.relative_to(experiments))] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return {"schema_version": 1, "files": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "archive"))
    parser.add_argument("--root", type=Path, default=Path("/lambda/nfs/scalar-pi"))
    parser.add_argument("--study", choices=("full", "pilot"), default="full")
    parser.add_argument("--diagnostic", action="store_true", help="Only four released-policy compatibility episodes")
    parser.add_argument("--attempt", default="default", help="Separate diagnostic attempts; never overwrite evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = (
        render_matrix(repo, args.root, args.study, diagnostic=args.diagnostic, attempt=args.attempt)
        if args.action == "render"
        else archive(args.root)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
