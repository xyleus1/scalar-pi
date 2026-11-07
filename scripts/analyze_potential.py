"""Validate a completed matrix before computing paired success-rate statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.research import evaluation


def analyze(manifest: dict, *, comparisons: list[str], resamples: int, random_seed: int) -> dict:
    records = evaluation.read_episodes([cell["episode_log_path"] for cell in evaluation.matrix_cells(manifest)])
    report = {
        "validation": evaluation.validate_results(records, manifest),
        "diagnostic": manifest.get("diagnostic", False),
        "summary": evaluation.summarize(records),
        "comparisons": [],
        "notes": [
            "Differences and intervals are fractions; multiply by 100 for percentage points.",
            "90% bootstrap CI within +/-0.03 is an equivalence criterion, not a TOST p-value.",
            "Latency includes all observed requests, including cold requests; not a warm steady-state benchmark.",
            "Released references are reported separately, not as independent training-seed replicates.",
        ],
    }

    def compare(left_group, left_k, right_group, right_k, kind):
        left = [r for r in records if r["group"] == left_group and r["num_steps"] == left_k]
        right = [r for r in records if r["group"] == right_group and r["num_steps"] == right_k]
        result = evaluation.paired_bootstrap(left, right, resamples=resamples, random_seed=random_seed)
        report["comparisons"].append(
            {
                "kind": kind,
                "left": {"group": left_group, "num_steps": left_k},
                "right": {"group": right_group, "num_steps": right_k},
                **result,
            }
        )

    for comparison in comparisons:
        parts = comparison.split(":")
        if len(parts) != 2 or not all(parts):
            raise ValueError("Comparisons use LEFT_GROUP:RIGHT_GROUP, e.g. P:B1")
        for k in manifest.get("num_steps", [1, 10]):
            compare(parts[0], k, parts[1], k, "architecture")
    if {1, 10}.issubset(manifest.get("num_steps", [1, 10])):
        for group in sorted({v["group"] for v in manifest["variants"]}):
            compare(group, 1, group, 10, "integration_steps")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--compare", action="append", default=[], help="Paired architecture contrast, e.g. P:B1")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output_dir = Path(manifest.get("output_dir", "results/potential"))
    if not output_dir.is_absolute():
        manifest["output_dir"] = str(Path(__file__).resolve().parents[1] / output_dir)
    report = analyze(manifest, comparisons=args.compare, resamples=args.resamples, random_seed=args.random_seed)
    result = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
