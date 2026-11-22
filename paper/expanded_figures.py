"""Regenerate all paper figures from the unchanged saved study report."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from architecture_figures import draw_all as draw_architectures
from figure_style import setup_style
from result_figures import draw_all as draw_results

HERE = Path(__file__).resolve().parent


REPORT = HERE.parent / "results/study-v0.1.0.json"


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


SEEDS = (17, 42, 93)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_names(dataset, previous):
    if dataset is None:
        return previous["task_descriptions"]
    index = json.loads((dataset / "dataset-index.json").read_bytes())
    names = {}
    for task in index["tasks"]:
        key = (task["suite"], task["task_id"])
        if key in names:
            continue
        path = dataset / task["task_log"]["path"]
        with path.open(encoding="utf8") as stream:
            row = json.loads(next(stream))
        names[key] = {
            "suite": key[0],
            "task_id": key[1],
            "description": row["task_description"],
            "public_log": task["task_log"]["path"],
            "public_log_sha256": task["task_log"]["sha256"],
        }
    assert len(names) == 40
    return sorted(names.values(), key=lambda t: (SUITES.index(t["suite"]), t["task_id"]))


def figure_data(dataset=None):
    path = HERE / "figure-data.json"
    previous = json.loads(path.read_bytes()) if path.exists() else {}
    report = json.loads(REPORT.read_bytes())
    summary = report["summary"]
    names = task_names(dataset, previous)
    by_task = {(r["checkpoint_id"], r["num_steps"], r["suite"], r["task_id"]): r for r in summary["tasks"]}
    assert len(by_task) == 800
    assert sum(r["episodes"] for r in summary["tasks"]) == 40000
    tasks = []
    for name in names:
        for k in (1, 10):
            rates = {
                g: [by_task[(f"{g}-seed{s}", k, name["suite"], name["task_id"])]["success_rate"] for s in SEEDS]
                for g in ("B0", "B1", "P")
            }
            ref = by_task[("released-pi05-libero", k, name["suite"], name["task_id"])]["success_rate"]
            means = {g: statistics.mean(v) for g, v in rates.items()}
            tasks.append(
                {
                    "suite": name["suite"],
                    "task_id": name["task_id"],
                    "description": name["description"],
                    "num_steps": k,
                    "seeds": list(SEEDS),
                    "seed_success_rates": rates,
                    "mean_success_rates": means,
                    "released_success_rate": ref,
                    "P_minus_B1_pp": 100 * (means["P"] - means["B1"]),
                    "P_minus_reference_pp": 100 * (means["P"] - ref),
                    "P_minus_B1_by_seed_pp": [100 * (p - b) for p, b in zip(rates["P"], rates["B1"], strict=True)],
                }
            )
    data = {
        "kind": "scalar_pi_expanded_figure_data_v1",
        "author": "Nima Kamali and Mert Turna",
        "license": "CC-BY-4.0",
        "source_report": {"path": "results/study-v0.1.0.json", "sha256": digest(REPORT)},
        "episode_count": 40000,
        "task_count": 800,
        "hardware_acceptance_claimed": False,
        "analysis_status": "descriptive_only_not_hardware_qualified",
        "new_statistics_estimated": False,
        "task_descriptions": names,
        "suite_success": summary["trained"] + summary["reference"],
        "overall_success": summary["checkpoint_macro_averages"],
        "contrasts": report["comparisons"],
        "task_comparisons": tasks,
        "interpretation": "Seed points are observed rates, not confidence intervals. Task case studies are exploratory. Success differences do not identify a mechanism or establish motion smoothness.",
        "captions": CAPTIONS,
        "literature_evidence": LITERATURE,
        "architecture_sources": {
            "direct_openpi": "https://github.com/Physical-Intelligence/openpi/tree/215abfb217dbac7d5f1273282331b9b1866c0479",
            "prior_input_dot": "https://arxiv.org/html/2608.18004v1",
            "energyflow_scalar_head": "https://arxiv.org/html/2605.00623v2",
            "scalar_pi": "src/openpi/models_pytorch/potential.py",
        },
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf8")
    return data


CAPTIONS = {
    "method": "Shared action expert and direct/scalar readouts. The scalar dot uses the complete noisy chunk; its full derivative retains f and the vector-Jacobian product. The dashed path denotes differentiation. The compact strip states the trained-arm and sampling settings.",
    "success_overview": "Observed success by suite and solver steps. Shapes and colors identify models. Within each trained arm, points run left to right for seeds 17, 42, 93 and short horizontal bars show their mean; the released reference has one checkpoint and therefore one point, not three seeds. Overall rates weight the four equally sized suites equally. No confidence intervals are constructed from these seed points.",
    "contrast_intervals": "Eight pre-specified descriptive paired contrasts copied exactly from the saved report. Points are percentage-point differences and bars are the original 95% percentile-bootstrap intervals (10,000 resamples, seed 0). The primary P-B1,K10 contrast is highlighted. Intervals are not adjusted for multiplicity and do not resolve the study's hardware-eligibility limitation.",
    "task_differences": "Exploratory task heterogeneity on the ten long-horizon tasks at K10. Points show trained-seed mean P minus B1 or the single released reference, in percentage points. The two selected K10 tasks reproduce the quoted Moka 53.3% versus 38% and mugs 64% versus 100% rates. Seed-level points expose variation; the horizontal released line is one checkpoint, not a repeated seed. Selection is descriptive and does not support a mechanistic smoothness explanation.",
}


LITERATURE = [
    {
        "name": "Hermite",
        "protocol": "LIBERO / pi0.5",
        "baseline": 95.9,
        "variant": 98.7,
        "baseline_label": "pi0.5",
        "variant_label": "+ Hermite",
        "status": "primary_verified",
        "source": "https://arxiv.org/html/2608.01265v2",
        "source_location": "Tables II and V",
        "protocol_detail": "Four-suite LIBERO average; 50 rollouts per task, 40 tasks, 2,000 episodes per method; T10/W5, ten denoising steps, 30,000 training updates.",
    },
    {
        "name": "Faithful Plans",
        "protocol": "Custom Block-AR comparison",
        "baseline": 95.85,
        "variant": 98.45,
        "baseline_label": "Suffix\nconditioning",
        "variant_label": "NGM",
        "status": "primary_verified",
        "source": "https://arxiv.org/html/2609.30833v1",
        "source_location": "Table 2",
        "protocol_detail": "Custom Block-AR + suffix conditioning versus Custom Block-AR + NGM; 500 episodes per suite; one training run per variant. Not a vanilla pi0.5 baseline.",
    },
    {
        "name": "ESP",
        "protocol": "Authors' flow comparison",
        "baseline": 89.1,
        "variant": 91.8,
        "baseline_label": "10-step flow",
        "variant_label": "1-step ESP",
        "status": "primary_verified",
        "source": "https://arxiv.org/html/2610.07696v1",
        "source_location": "Figure 4",
        "protocol_detail": "Authors' ten-step flow baseline versus one-step ESP; reported values 89.1 +/- 1.2 and 91.8 +/- 1.4. The wide ESP variant in Table II is a separate configuration, not this plotted comparison.",
    },
    {
        "name": "Drifting",
        "protocol": "Full-model pi0.5",
        "baseline": 97.10,
        "variant": 98.32,
        "baseline_label": "pi0.5",
        "variant_label": "Drifting",
        "status": "primary_verified",
        "source": "https://arxiv.org/html/2608.29749v1",
        "protocol_detail": "Full-model updates with a fresh head; pi0.5 97.10 versus drifting 98.32. Frozen-VLM expert-only drifting is a different setting and reports 94.57, not the plotted 98.32.",
    },
    {
        "name": "SAVLA",
        "protocol": "10% GR00T data",
        "baseline": 48.3,
        "variant": 57.1,
        "baseline_label": "Reduced\nGR00T N1.5",
        "variant_label": "SAVLA",
        "status": "primary_verified",
        "source": "https://arxiv.org/html/2609.16641v1",
        "source_location": "Table V and section V-D",
        "protocol_detail": "Identical ten-percent per-task data; budget-matched reduced GR00T N1.5 baseline; 200 evaluation episodes, three seeds.",
    },
    {
        "name": "AnyCam",
        "protocol": "pi0.5 / unseen agent camera",
        "baseline": 69.9,
        "intermediate": 87.2,
        "intermediate_label": "View augmentation",
        "variant": 94.5,
        "baseline_label": "pi0.5",
        "variant_label": "AnyCam",
        "status": "primary_verified",
        "source": "https://arxiv.org/html/2603.05868v3",
        "source_location": "Table I",
        "protocol_detail": "Unseen agent-camera perturbations, small/medium/large levels, averaged across four suites. pi0.5, view-augmented pi0.5, and AnyCam + frozen pi0.5. Different from standard LIBERO camera evaluation.",
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Public dataset folder; only needed to refresh task descriptions")
    args = parser.parse_args()
    setup_style()
    data = figure_data(args.dataset)
    draw_architectures(data)
    draw_results(data)
    print(
        json.dumps(
            {
                "figures": 10,
                "source_report_sha256": data["source_report"]["sha256"],
                "episode_count": data["episode_count"],
                "task_count": data["task_count"],
                "new_statistics_estimated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
