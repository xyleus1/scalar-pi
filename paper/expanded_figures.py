"""Draw original study figures from saved results; no new statistical estimation.

Use --dataset once to source task descriptions from the public release. Subsequent
builds use those descriptions and their source hashes in figure-data.json.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import textwrap

import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent / "results/study-v0.1.0.json"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_LABELS = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long horizon",
}
SEEDS = (17, 42, 93)
COLORS = {"B0": "#3B648D", "B1": "#D27A35", "P": "#087F83", "reference": "#343D48"}
INK, MUTED, GRID = "#243243", "#617083", "#E1E6EB"
LONG_TASK_LABELS = [
    "Soup + tomato sauce to basket",
    "Cream cheese + butter to basket",
    "Stove on + Moka pot",
    "Bowl to bottom drawer; close",
    "Two mugs to separate plates",
    "Book to caddy back compartment",
    "Mug + pudding to plate",
    "Soup + cream cheese to basket",
    "Both Moka pots to stove",
    "Mug to microwave; close",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": True,
        }
    )


def save(fig, name):
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(
        folder / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Author": "Nima Kamali and Mert Turna", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(folder / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def box(ax, xy, size, text, *, fill="#F2F5F8", edge=GRID, fontsize=10, weight="normal", dashed=False):
    x, y = xy
    w, h = size
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.055",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1,
            linestyle="--" if dashed else "-",
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, linespacing=1.45
    )


def arrow(ax, start, end, color=MUTED):
    ax.annotate(
        "", end, start, arrowprops={"arrowstyle": "-|>", "lw": 1.15, "color": color, "shrinkA": 3, "shrinkB": 3}
    )


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
    "comparison_map": "Original architectural comparison. OpenPI predicts a direct action velocity; Scalar-Pi retains its transformer vector output but differentiates an input-dot scalar. The input-dot construction also appears in a U-Net setting in Sun et al.; EnergyFlow instead pools temporal features into a scalar head and uses a negative action gradient as a score. These are different implementations and objectives, not a controlled empirical comparison. The scalar-potential construction itself is not claimed as new.",
    "study_design": "Factorial protocol: three trained arms at three seeds, one released reference, two Euler-step counts, four suites of ten tasks, and fifty initial states per task. Nine trained checkpoints contribute 36,000 episodes and the single released checkpoint contributes 4,000. B1 and P share reset output initialization; B0 retains the pretrained output. Hardware eligibility remains unresolved.",
    "success_overview": "Observed success by suite and solver steps. For trained arms, shapes mark seeds 17, 42, 93 and short horizontal bars show their mean; the released reference has one checkpoint and therefore one point, not three seeds. Overall rates weight the four equally sized suites equally. No confidence intervals are constructed from these seed points.",
    "contrast_intervals": "Eight pre-specified descriptive paired contrasts copied exactly from the saved report. Points are percentage-point differences and bars are the original 95% percentile-bootstrap intervals (10,000 resamples, seed 0). The primary P-B1,K10 contrast is highlighted. Intervals are not adjusted for multiplicity and do not resolve the study's hardware-eligibility limitation.",
    "task_differences": "Exploratory task heterogeneity on the ten long-horizon tasks. Heatmap cells show trained-seed mean P minus B1 or the single released reference, in percentage points. The two selected K10 tasks reproduce the quoted Moka 53.3% versus 38% and mugs 64% versus 100% rates. Seed-level points expose variation; the horizontal released line is one checkpoint, not a repeated seed. Selection is descriptive and does not support a mechanistic smoothness explanation.",
    "task_heatmap_all": "All forty benchmark tasks, with P minus B1 and P minus the released checkpoint at both solver settings. Trained-arm rates are averages over three seeds and fifty initial states per seed; reference rates use one checkpoint and fifty initial states. Cells show percentage-point differences, not task-level confidence intervals. Task labels are abbreviated from the public episode descriptions; the complete descriptions and numeric cells are in figure-data.json.",
    "future_studies": "Proposed studies only: training-time trajectory supervision, event/waypoint conditioning, invariant-potential geometry, future-feature prediction, and a discrete-gripper/convex-noise generator. None is evaluated in this report. Every proposal requires matched direct/scalar controls and task-success evaluation; smoothness, safety and distributional coverage require their own measured endpoints.",
    "literature_evidence": "Within-paper results reported by other studies, redrawn from their numerical reports. Each panel uses its own task/data/model protocol; gains are not pooled and the panels are not a cross-paper leaderboard. There is no Scalar-Pi bar. These results motivate controlled follow-up experiments rather than predict the effect of adding a component here.",
}

# Numerical values are supplied from the primary-source audit; protocol labels
# intentionally prevent interpreting the panels as one shared benchmark.
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


def comparison_map():
    fig, ax = plt.subplots(figsize=(11.4, 4.05))
    ax.set(xlim=(0, 12), ylim=(0, 4.2))
    ax.axis("off")
    titles = ["Direct OpenPI", "Prior input-dot model", "Scalar-Pi (this study)", "EnergyFlow"]
    notes = ["Direct action field", "Sun et al. / U-Net", "Original pi0.5 transformer", "Temporal U-Net / score model"]
    trunks = [
        "Transformer\nvector output",
        "U-Net\nvector output",
        "Transformer\nvector output",
        "Temporal features\n+ global pooling",
    ]
    middle = [
        r"$v=f_\theta(x,t,c)$",
        r"$\Phi=x^\top f_\theta(x,t,c)$",
        r"$\Psi=x^\top f_\theta(x,t,c)$",
        "Scalar MLP head\n" + r"$E_\theta(x,t,c)$",
    ]
    outputs = ["Direct velocity", r"$\nabla_x\Phi$", r"$v=\nabla_x\Psi$", r"$s=-\nabla_x E_\theta$"]
    for i in range(4):
        x = 0.12 + 3 * i
        col = COLORS["P"] if i == 2 else INK
        ax.text(x + 1.35, 3.95, titles[i], ha="center", fontsize=11, weight="bold", color=col)
        ax.text(x + 1.35, 3.65, notes[i], ha="center", fontsize=8, color=MUTED)
        box(ax, (x, 2.65), (2.7, 0.67), trunks[i], fill="#E8F2F2" if i == 2 else "#F2F5F8")
        arrow(ax, (x + 1.35, 2.62), (x + 1.35, 2.34), col)
        box(ax, (x, 1.53), (2.7, 0.77), middle[i], fill="#E8F2F2" if i == 2 else "#F2F5F8", fontsize=11)
        arrow(ax, (x + 1.35, 1.51), (x + 1.35, 1.19), col)
        box(ax, (x, 0.52), (2.7, 0.63), outputs[i], edge=col, fontsize=11)
    ax.text(
        6,
        0.12,
        "Architectural relationships only. Different hosts, training objectives, time conventions and gradient signs.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "comparison_map")


def study_design():
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    ax.text(0.12, 4.72, "TRAIN", weight="bold", fontsize=10, color=MUTED)
    arms = [
        ("B0", "Pretrained output", COLORS["B0"]),
        ("B1", "Reset direct output", COLORS["B1"]),
        ("P", "Reset dot-potential", COLORS["P"]),
    ]
    for j, (arm, label, col) in enumerate(arms):
        y = 3.65 - j * 0.97
        box(ax, (0.12, y), (2.65, 0.72), arm + "  |  " + label, edge=col, fontsize=10)
        arrow(ax, (2.82, y + 0.36), (3.22, y + 0.36), col)
        box(ax, (3.25, y), (2.4, 0.72), "Seeds 17 / 42 / 93\n30,000 updates each", fontsize=9.5)
    ax.text(
        2.86,
        1.25,
        "Same frozen VLM; matched data and optimizer.\nB1 and P start from identical reset tensors.",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    box(
        ax,
        (0.12, 0.16),
        (5.53, 0.66),
        "Released pi0.5 checkpoint  |  no study training",
        fill="#F7F7F7",
        edge=COLORS["reference"],
        fontsize=10,
    )
    ax.text(6.65, 4.72, "EVALUATE", weight="bold", fontsize=10, color=MUTED)
    box(ax, (6.65, 3.38), (5.17, 0.92), "10 checkpoints x 2 solver settings\nK = 1 and K = 10 Euler steps", fontsize=11)
    for y in (4.01, 3.04, 2.07):
        ax.plot([5.7, 5.98], [y, y], color=MUTED, lw=1)
    ax.plot([5.98, 5.98], [2.07, 4.01], color=MUTED, lw=1)
    arrow(ax, (5.98, 3.04), (6.55, 3.79))
    arrow(ax, (5.72, 0.49), (6.55, 3.58))
    arrow(ax, (9.23, 3.34), (9.23, 2.92))
    box(
        ax,
        (6.65, 1.94),
        (5.17, 0.91),
        "4 suites x 10 tasks x 50 initial states\nOriginal fixed evaluation identities",
        fontsize=11,
    )
    arrow(ax, (9.23, 1.9), (9.23, 1.49))
    box(
        ax,
        (6.65, 0.48),
        (5.17, 0.94),
        "40,000 episodes  |  800 task evaluations\n36,000 trained + 4,000 released-reference",
        fill="#E8F2F2",
        edge=COLORS["P"],
        fontsize=11,
        weight="bold",
    )
    ax.text(
        9.23, 0.12, "Task-validated outcomes; hardware eligibility unresolved.", ha="center", fontsize=8.5, color=MUTED
    )
    save(fig, "study_design")


def success_overview(data):
    fig, axes = plt.subplots(2, 5, figsize=(11.4, 5.4), sharey=True)
    suite_rows = data["suite_success"]
    overall = data["overall_success"]
    markers = ("o", "^", "s")
    for row, k in enumerate((1, 10)):
        for col, suite in enumerate((*SUITES, "overall")):
            ax = axes[row, col]
            rows = [
                r
                for r in (overall if suite == "overall" else suite_rows)
                if r["num_steps"] == k and (suite == "overall" or r["suite"] == suite)
            ]
            for x, g in enumerate(("B0", "B1", "P", "reference")):
                selected = [r for r in rows if r["group"] == g]
                values = [100 * r["success_rate"] for r in selected]
                if g == "reference":
                    assert len(values) == 1
                    ax.scatter([x], values, color=COLORS[g], marker="D", s=24, zorder=4)
                else:
                    ordered = sorted(selected, key=lambda r: r["training_seed"])
                    for j, r in enumerate(ordered):
                        ax.scatter(
                            x + (j - 1) * 0.14,
                            100 * r["success_rate"],
                            marker=markers[j],
                            color=COLORS[g],
                            s=20,
                            edgecolors="white",
                            linewidths=0.35,
                            zorder=4,
                        )
                    ax.plot([x - 0.28, x + 0.28], [statistics.mean(values)] * 2, color=COLORS[g], lw=2, zorder=3)
            ax.set(
                ylim=(70, 101.5),
                xlim=(-0.55, 3.55),
                xticks=range(4),
                xticklabels=["B0", "B1", "P", "Ref."],
                yticks=[70, 80, 90, 100],
            )
            ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(SUITE_LABELS.get(suite, "Overall"), fontweight="bold", pad=10)
            if col == 0:
                ax.set_ylabel(f"K = {k}\nSuccess (%)")
    handles = [
        Line2D([], [], color=MUTED, marker=m, ls="", markersize=4, label=f"Seed {s}")
        for m, s in zip(markers, SEEDS, strict=True)
    ]
    handles += [
        Line2D([], [], color=MUTED, lw=2, label="Trained-arm mean"),
        Line2D([], [], color=COLORS["reference"], marker="D", ls="", markersize=4, label="Released checkpoint"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=5, frameon=False, fontsize=8.5)
    fig.subplots_adjust(wspace=0.2, hspace=0.32, bottom=0.15, top=0.92, left=0.065, right=0.99)
    save(fig, "success_overview")


def contrast_intervals(data):
    fig, ax = plt.subplots(figsize=(10.7, 4.9))
    labels = []
    for i, c in enumerate(data["contrasts"]):
        left, right = c["left"], c["right"]
        primary = c["endpoint"] == "primary"

        def g(value):
            return "Released" if value == "reference" else value

        label = (
            f"{g(left['group'])} - {g(right['group'])}  |  K = {left['num_steps']}"
            if c["kind"] == "architecture"
            else f"{g(left['group'])}  |  K = 1 - K = 10"
        )
        labels.append(label + ("  [primary]" if primary else ""))
        y = 7 - i
        point = 100 * c["difference_left_minus_right"]
        lo, hi = [100 * v for v in c["ci95"]]
        color = COLORS["P"] if primary else COLORS["B0"]
        if primary:
            ax.axhspan(y - 0.4, y + 0.4, color="#E8F2F2", zorder=0)
        ax.plot([lo, hi], [y, y], color=color, lw=2.1 if primary else 1.5)
        ax.plot([lo, hi], [y, y], linestyle="", marker="|", color=color, markersize=7)
        ax.scatter([point], [y], color=color, s=39 if primary else 23, zorder=3)
        ax.text(
            4.05,
            y,
            f"{point:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold" if primary else "normal",
        )
    ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=0)
    ax.axhline(3.5, color=GRID, lw=0.8)
    ax.set(
        yticks=range(8),
        yticklabels=labels[::-1],
        xlim=(-2.25, 7),
        ylim=(-0.7, 7.7),
        xticks=[-2, -1, 0, 1, 2, 3],
        xlabel="Difference in success (percentage points)",
    )
    ax.text(4.05, 8.02, "Difference  [95% interval]", fontsize=8.5, color=MUTED)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.265, right=0.99, bottom=0.13, top=0.91)
    save(fig, "contrast_intervals")


def task_differences(data):
    rows = {(r["num_steps"], r["task_id"]): r for r in data["task_comparisons"] if r["suite"] == "libero_10"}
    matrix = np.array(
        [
            [rows[(k, t)][field] for field in ("P_minus_B1_pp", "P_minus_reference_pp") for k in (1, 10)]
            for t in range(10)
        ]
    )
    labels = [
        "00  Soup + tomato sauce → basket",
        "01  Cream cheese + butter → basket",
        "02  Stove on + Moka pot",
        "03  Bowl → bottom drawer; close",
        "04  Two mugs → separate plates",
        "05  Book → caddy back compartment",
        "06  Mug + pudding → plate",
        "07  Soup + cream cheese → basket",
        "08  Both Moka pots → stove",
        "09  Mug → microwave; close",
    ]
    fig = plt.figure(figsize=(11.4, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1], wspace=0.52, hspace=0.58)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.pcolormesh(
        np.arange(5) - 0.5,
        np.arange(11) - 0.5,
        matrix,
        cmap="RdBu",
        vmin=-40,
        vmax=40,
        shading="flat",
        rasterized=False,
    )
    ax.set(xlim=(-0.5, 3.5), ylim=(9.5, -0.5))
    ax.set(yticks=range(10), yticklabels=labels, xticks=range(4), xticklabels=["K1", "K10", "K1", "K10"])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.text(0.5, 1.085, "P - B1", transform=ax.get_xaxis_transform(), ha="center", fontsize=10, weight="bold")
    ax.text(2.5, 1.085, "P - released", transform=ax.get_xaxis_transform(), ha="center", fontsize=10, weight="bold")
    ax.axvline(1.5, color="white", lw=3)
    for y in range(10):
        for x in range(4):
            ax.text(
                x,
                y,
                f"{matrix[y,x]:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(matrix[y, x]) > 24 else INK,
            )
    for y in (3.5, 4.5, 7.5, 8.5):
        ax.axhline(y, color="white", lw=0.8)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.065, aspect=30)
    cb.solids.set_rasterized(False)
    cb.set_label("Success difference (percentage points)", fontsize=8)
    cb.set_ticks([-40, -20, 0, 20, 40])
    cb.ax.tick_params(labelsize=8)
    for j, (task, title) in enumerate(((8, "Both Moka pots → stove"), (4, "Two mugs → separate plates"))):
        a = fig.add_subplot(gs[j, 1])
        r = rows[(10, task)]
        for g in ("B1", "P"):
            vals = np.array(r["seed_success_rates"][g]) * 100
            a.plot(
                [0, 1, 2],
                vals,
                color=COLORS[g],
                marker="o",
                ms=4,
                ls="",
                label=f"{g}: mean {statistics.mean(vals):.1f}%",
            )
        ref = r["released_success_rate"] * 100
        a.axhline(ref, color=COLORS["reference"], ls="--", lw=1.2, label=f"Released: {ref:.0f}%")
        a.set(
            title=f"{title}  |  K10",
            xticks=[0, 1, 2],
            xticklabels=[str(s) for s in SEEDS],
            ylim=(15, 105),
            yticks=[20, 40, 60, 80, 100],
            ylabel="Success (%)",
            xlabel="Training seed",
        )
        a.grid(axis="y", color=GRID, lw=0.5)
        a.legend(loc="lower right" if task == 4 else "upper right", fontsize=7.5, frameon=False)
    fig.subplots_adjust(left=0.31, right=0.99, bottom=0.14, top=0.88)
    save(fig, "task_differences")


def future_studies():
    fig, ax = plt.subplots(figsize=(11.4, 6.25))
    ax.set(xlim=(0, 12), ylim=(0, 6.4))
    ax.axis("off")
    proposals = [
        (
            "01  Trajectory supervision",
            "Training-only Hermite auxiliary head",
            "Two cubic segments x 6 arm controls x 10 steps\nFlow loss + 10 x reconstruction; gripper unchanged",
            "Direct/scalar x Hermite/cosine/no head",
        ),
        (
            "02  Event / waypoint conditioning",
            "State + short history → event prediction",
            "Predict gripper event, pose and time-to-event;\ncondition the action expert on that prediction",
            "Matched state-only, history and waypoint controls",
        ),
        (
            "03  Geometric structure",
            "Invariant scalar → equivariant gradient",
            r"$\Psi(Rx,t,T_Rc)=\Psi(x,t,c)$ for orthogonal $R$"
            "\nCalibrated relative coordinates; explicit symmetries",
            "Matched augmentation and direct-equivariant models",
        ),
        (
            "04  Future-feature prediction",
            "Training-only visual-change auxiliary head",
            "Current features + 5 demonstration actions\n→ future visual-feature change; frozen targets",
            "Current / future / shuffled target controls",
        ),
        (
            "05  Distributional action generation",
            "Discrete gripper + convex noise potential",
            r"$G$ categorical; $\nabla_z\Phi(z,c,G)$ → 60 arm commands"
            "\nEnergy-score training; not the current velocity field",
            "Matched direct generator; distribution + task endpoints",
        ),
    ]
    ax.text(0.12, 6.14, "PROPOSED ONLY  /  NO RESULTS IN THIS STUDY", weight="bold", fontsize=11, color=COLORS["P"])
    for j, (title, head, body, control) in enumerate(proposals):
        y = 4.91 - j * 1.11
        box(
            ax,
            (0.12, y),
            (2.5, 0.91),
            textwrap.fill(title, 23),
            fill="#E8F2F2",
            edge=COLORS["P"],
            fontsize=9.2,
            weight="bold",
        )
        arrow(ax, (2.66, y + 0.46), (3.04, y + 0.46), COLORS["P"])
        box(ax, (3.08, y), (5.1, 0.91), head + "\n" + body, fontsize=8.7)
        box(ax, (8.4, y), (3.42, 0.91), "CONTROL / TEST\n" + textwrap.fill(control, 37), fontsize=8.3, dashed=True)
    ax.text(
        6,
        0.03,
        "A structural constraint is a hypothesis, not a guarantee of smoothness, safety, multimodality or better task success.",
        ha="center",
        fontsize=8.3,
        color=MUTED,
    )
    save(fig, "future_studies")


def literature_evidence():
    if any(r["status"] != "primary_verified" for r in LITERATURE):
        return
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.7))
    for ax, r in zip(axes.flat, LITERATURE, strict=True):
        start, end = r["baseline"], r["variant"]
        low = max(0, min(start, end) - 5)
        high = min(101, max(start, end) + 4)
        values = [start, end]
        labels = [r["baseline_label"], r["variant_label"]]
        if "intermediate" in r:
            values.insert(1, r["intermediate"])
            labels.insert(1, "View\naugmentation")
        positions = list(range(len(values)))
        ax.plot(positions, values, color=COLORS["P"], lw=1.5)
        ax.scatter([0], [start], color=COLORS["B0"], s=38, zorder=3)
        ax.scatter(positions[1:], values[1:], color=COLORS["P"], s=38, zorder=3)
        for x, y in zip(positions, values, strict=True):
            ax.annotate(
                f"{y:g}%", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", weight="bold", fontsize=10
            )
        ax.set(
            xticks=positions,
            xticklabels=labels,
            xlim=(-0.35, positions[-1] + 0.35),
            ylim=(low, high),
            ylabel="Success (%)",
        )
        ax.set_title(r["name"] + "\n" + r["protocol"], fontsize=10, pad=18)
        delta = f"{end-start:+.2f} pp"
        if "intermediate" in r:
            delta = f"Augmentation +{r['intermediate']-start:.1f} pp; AnyCam +{end-r['intermediate']:.1f} pp further"
        ax.text(
            0.5, 0.04, delta, transform=ax.transAxes, ha="center", fontsize=8 if len(values) > 2 else 9, color=MUTED
        )
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("OTHER STUDIES  /  WITHIN-PAPER COMPARISONS ONLY", fontsize=11, weight="bold", color=MUTED, y=1.015)
    fig.subplots_adjust(wspace=0.34, hspace=0.65, left=0.06, right=0.99, bottom=0.07, top=0.86)
    save(fig, "literature_evidence")


def task_heatmap_all(data):
    rows = {(r["suite"], r["num_steps"], r["task_id"]): r for r in data["task_comparisons"]}
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.3))
    for ax, suite in zip(axes.flat, SUITES, strict=True):
        values = np.array(
            [
                [rows[(suite, k, task)][field] for field in ("P_minus_B1_pp", "P_minus_reference_pp") for k in (1, 10)]
                for task in range(10)
            ]
        )
        labels = []
        for task in range(10):
            label = rows[(suite, 1, task)]["description"]
            label = label.replace("pick up the black bowl", "bowl").replace("pick up the ", "")
            label = label.replace(" and place it on the plate", " to plate").replace(
                " and place it in the basket", " to basket"
            )
            label = label.replace("put the ", "").replace(" and put the ", "; ")
            if suite == "libero_10":
                label = LONG_TASK_LABELS[task]
            labels.append(f"{task:02d}  " + textwrap.fill(label, 30))
        im = ax.pcolormesh(
            np.arange(5) - 0.5,
            np.arange(11) - 0.5,
            values,
            cmap="RdBu",
            vmin=-40,
            vmax=40,
            shading="flat",
            rasterized=False,
        )
        ax.set(xlim=(-0.5, 3.5), ylim=(9.5, -0.5))
        ax.set(yticks=range(10), yticklabels=labels, xticks=range(4), xticklabels=["K1", "K10", "K1", "K10"])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, labelsize=7.7)
        ax.text(0.5, 1.07, "P - B1", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.text(2.5, 1.07, "P - released", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        ax.set_title(SUITE_LABELS[suite], pad=42, fontweight="bold", fontsize=11)
        ax.axvline(1.5, color="white", lw=3)
        for y in range(10):
            for x in range(4):
                ax.text(
                    x,
                    y,
                    f"{values[y,x]:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(values[y, x]) > 24 else INK,
                )
    fig.subplots_adjust(left=0.22, right=0.99, top=0.92, bottom=0.13, wspace=1.05, hspace=0.34)
    cax = fig.add_axes((0.3, 0.055, 0.6, 0.018))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.solids.set_rasterized(False)
    cb.set_ticks([-40, -20, 0, 20, 40])
    cb.set_label("Success difference (percentage points)")
    save(fig, "task_heatmap_all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, help="Extracted public data folder, only needed to refresh task descriptions"
    )
    args = parser.parse_args()
    style()
    data = figure_data(args.dataset)
    comparison_map()
    study_design()
    success_overview(data)
    contrast_intervals(data)
    task_differences(data)
    future_studies()
    literature_evidence()
    task_heatmap_all(data)
    print(
        json.dumps(
            {
                "figures": 7 + int(all(r["status"] == "primary_verified" for r in LITERATURE)),
                "source_report_sha256": data["source_report"]["sha256"],
                "episode_count": data["episode_count"],
                "task_count": data["task_count"],
                "new_statistics_estimated": False,
            }
        )
    )


if __name__ == "__main__":
    main()
