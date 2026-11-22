"""Publication figures from the retained study arrays, without new estimation."""

import json
from pathlib import Path

from figure_style import COLORS
from figure_style import DIFFERENCE_CMAP
from figure_style import GRID
from figure_style import INK
from figure_style import MUTED
from figure_style import save
from figure_style import style
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_LABELS = ("Spatial", "Object", "Goal", "Long", "Overall")
GROUPS = ("B0", "B1", "P", "reference")
SEEDS = (17, 42, 93)
MARKERS = {"B0": "s", "B1": "^", "P": "o", "reference": "D"}
NAMES = {"B0": "B0: pretrained direct", "B1": "B1: reset direct", "P": "P: scalar", "reference": "Released"}
TASK_LABELS = {
    "libero_spatial": [
        "Between plate / ramekin",
        "Next to ramekin",
        "Table center",
        "On cookie box",
        "In top drawer",
        "On ramekin",
        "Next to cookie box",
        "On stove",
        "Next to plate",
        "On cabinet",
    ],
    "libero_object": [
        "Alphabet soup",
        "Cream cheese",
        "Salad dressing",
        "BBQ sauce",
        "Ketchup",
        "Tomato sauce",
        "Butter",
        "Milk",
        "Chocolate pudding",
        "Orange juice",
    ],
    "libero_goal": [
        "Open middle drawer",
        "Bowl to stove",
        "Wine bottle to cabinet",
        "Bowl into top drawer",
        "Bowl to cabinet",
        "Plate to stove front",
        "Cream cheese in bowl",
        "Stove on",
        "Bowl to plate",
        "Wine bottle to rack",
    ],
    "libero_10": [
        "Soup + sauce to basket",
        "Cheese + butter to basket",
        "Stove on + moka pot",
        "Bowl in drawer; close",
        "Mugs to separate plates",
        "Book to caddy",
        "Mug + pudding to plate",
        "Soup + cheese to basket",
        "Both moka pots to stove",
        "Mug in microwave; close",
    ],
}


def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", fontsize=9, fontweight="normal", pad=8)


def tidy(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(length=2.5, width=0.5, pad=3, colors=INK)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.45, zorder=0)
        ax.set_axisbelow(True)


def model_legend(fig, *, anchor=(0.5, 1.0), groups=GROUPS):
    handles = [Line2D([], [], color=COLORS[g], marker=MARKERS[g], lw=0, markersize=4, label=NAMES[g]) for g in groups]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=anchor,
        ncol=len(groups),
        frameon=False,
        handletextpad=0.45,
        columnspacing=1.5,
        fontsize=8,
    )


def success_overview(data):
    fig, axes = plt.subplots(2, 1, figsize=(7, 3.9), sharex=True, sharey=True)
    for ax, k, letter in zip(axes, (1, 10), "ab", strict=True):
        for suite_index, suite in enumerate((*SUITES, "overall")):
            rows = [
                r
                for r in (data["overall_success"] if suite == "overall" else data["suite_success"])
                if r["num_steps"] == k and (suite == "overall" or r["suite"] == suite)
            ]
            for offset, group in zip((-0.27, -0.09, 0.09, 0.27), GROUPS, strict=True):
                selected = sorted([r for r in rows if r["group"] == group], key=lambda r: r.get("training_seed") or 0)
                vals = np.array([r["success_rate"] * 100 for r in selected])
                assert len(vals) == (1 if group == "reference" else 3)
                x = suite_index + offset
                if group == "reference":
                    ax.plot(x, vals[0], "D", color=COLORS[group], ms=4, markeredgewidth=0, zorder=4)
                else:
                    ax.scatter(
                        x + np.array([-0.035, 0, 0.035]),
                        vals,
                        s=11,
                        marker=MARKERS[group],
                        color=COLORS[group],
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=4,
                    )
                    ax.plot([x - 0.065, x + 0.065], [vals.mean()] * 2, color=COLORS[group], lw=1.5, zorder=3)
        panel(ax, letter, f"{k} Euler step" + ("s" if k != 1 else ""))
        ax.set(ylabel="Success (%)", ylim=(70, 101.5), yticks=[70, 80, 90, 100], xlim=(-0.55, 4.55))
        ax.axvline(3.5, color="#BBBBBB", lw=0.55, ls=(0, (2, 2)))
        tidy(ax)
    axes[-1].set(xticks=np.arange(5), xticklabels=SUITE_LABELS)
    axes[0].tick_params(labelbottom=True)
    model_legend(fig, anchor=(0.54, 1.005))
    fig.subplots_adjust(left=0.075, right=0.99, top=0.875, bottom=0.09, hspace=0.48)
    save(fig, "success_overview")


def contrast_intervals(data):
    records = list(data["contrasts"])
    architecture = sorted(
        [r for r in records if r["kind"] == "architecture"],
        key=lambda r: (r["endpoint"] != "primary", r["right"]["group"], r["left"]["num_steps"]),
    )
    solver = [r for r in records if r["kind"] != "architecture"]
    ordered = architecture + solver
    assert len(ordered) == 8
    fig = plt.figure(figsize=(7, 3.3))
    ax = fig.add_axes((0.35, 0.17, 0.39, 0.72))
    label_ax = fig.add_axes((0.02, 0.17, 0.32, 0.72), sharey=ax)
    value_ax = fig.add_axes((0.765, 0.17, 0.225, 0.72), sharey=ax)
    for text_ax in (label_ax, value_ax):
        text_ax.set_xlim(0, 1)
        text_ax.axis("off")
    for i, r in enumerate(ordered):
        y = 7 - i
        p = 100 * r["difference_left_minus_right"]
        lo, hi = np.array(r["ci95"]) * 100
        left, right = r["left"], r["right"]
        primary = r["endpoint"] == "primary"
        color = COLORS[left["group"]]
        group = "Released" if left["group"] == "reference" else left["group"]
        if r["kind"] == "architecture":
            label = f"P - {right['group']}, K = {left['num_steps']}"
        else:
            label = f"{group}: K = {left['num_steps']} - K = {right['num_steps']}"
        if primary:
            label += "  (primary)"
        label_ax.text(0, y, label, va="center", fontsize=8.3, color=color, fontweight="bold" if primary else "normal")
        value_ax.text(
            1,
            y,
            f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]",
            ha="right",
            va="center",
            fontsize=7.5,
            color=color,
        )
        ax.errorbar(
            p,
            y,
            xerr=[[p - lo], [hi - p]],
            fmt="o",
            color=color,
            markersize=4,
            elinewidth=1,
            capsize=2,
            capthick=0.8,
            zorder=3,
        )
    for a in (ax, label_ax, value_ax):
        a.set_ylim(-0.6, 7.6)
        a.axhline(3.5, lw=0.5, color="#BBBBBB", zorder=0)
    ax.axvline(0, color="#777777", lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.set(xlim=(-2, 4), xticks=[-2, -1, 0, 1, 2, 3, 4], yticks=[], xlabel="Success difference (pp)")
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    label_ax.text(0, 8.05, "Comparison", fontsize=8.2)
    value_ax.text(1, 8.05, "Difference [95% interval]", ha="right", fontsize=8.2)
    fig.text(
        0.02,
        0.025,
        "Paired episodes: 6,000 per trained contrast; 2,000 for the released policy.",
        fontsize=7.5,
        color=MUTED,
    )
    save(fig, "contrast_intervals")


def task_differences(data):
    rows = {(r["num_steps"], r["task_id"]): r for r in data["task_comparisons"] if r["suite"] == "libero_10"}
    fig = plt.figure(figsize=(7, 3.9))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.25, 1], hspace=0.58, wspace=0.52, left=0.255, right=0.975, bottom=0.2, top=0.87
    )
    ax = fig.add_subplot(gs[:, 0])
    for t in range(10):
        y = 9 - t
        r = rows[(10, t)]
        ax.plot(r["P_minus_B1_pp"], y + 0.1, "o", color=COLORS["P"], ms=3.6)
        ax.plot(r["P_minus_reference_pp"], y - 0.1, "D", color=COLORS["reference"], ms=3.2)
    ax.axvline(0, color="#888888", lw=0.6, ls=(0, (3, 3)))
    ax.set(
        yticks=np.arange(10),
        yticklabels=[f"{t:02d}  {TASK_LABELS['libero_10'][t]}" for t in range(9, -1, -1)],
        ylim=(-0.65, 9.65),
        xlim=(-40, 40),
        xticks=[-40, -20, 0, 20, 40],
        xlabel="Success difference (pp)",
    )
    ax.tick_params(axis="y", labelsize=7.5, length=0)
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    panel(ax, "a", "Long tasks, K = 10")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=COLORS["P"], ms=3, label="P - B1"),
            Line2D([], [], marker="D", ls="", color=COLORS["reference"], ms=3, label="P - released"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.48, -0.17),
        frameon=False,
        fontsize=7.5,
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    for index, (task, title, letter) in enumerate(((8, "Both moka pots", "b"), (4, "Two mugs", "c"))):
        a = fig.add_subplot(gs[index, 1])
        r = rows[(10, task)]
        for g, offset, marker in (("B1", -0.08, "^"), ("P", 0.08, "o")):
            vals = np.array(r["seed_success_rates"][g]) * 100
            a.plot(np.arange(3) + offset, vals, marker=marker, color=COLORS[g], linestyle="", ms=4)
        a.axhline(r["released_success_rate"] * 100, ls=(0, (3, 2)), lw=0.8, color=COLORS["reference"])
        a.set(xticks=np.arange(3), xticklabels=SEEDS, xlim=(-0.5, 2.5), ylim=(20, 105), yticks=[20, 60, 100])
        a.text(
            2.4,
            r["released_success_rate"] * 100 - 7,
            f"{r['released_success_rate'] * 100:.0f}%",
            ha="right",
            va="top",
            fontsize=7.5,
            color=COLORS["reference"],
        )
        a.set_ylabel("Success (%)", labelpad=2)
        if index == 1:
            a.set_xlabel("Training seed")
        tidy(a)
        panel(a, letter, title)
    handles = [
        Line2D([], [], marker="^", linestyle="", color=COLORS["B1"], ms=4, label="B1"),
        Line2D([], [], marker="o", linestyle="", color=COLORS["P"], ms=4, label="P"),
        Line2D([], [], ls=(0, (3, 2)), color=COLORS["reference"], lw=0.8, label="Released"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.0),
        frameon=False,
        ncol=3,
        fontsize=7.6,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    save(fig, "task_differences")


def task_heatmap_all(data):
    rows = {(r["suite"], r["num_steps"], r["task_id"]): r for r in data["task_comparisons"]}
    fig = plt.figure(figsize=(7, 7.1))
    gs = fig.add_gridspec(2, 2, left=0.23, right=0.985, top=0.91, bottom=0.14, wspace=1.18, hspace=0.36)
    for index, suite in enumerate(SUITES):
        ax = fig.add_subplot(gs[index // 2, index % 2])
        values = np.array(
            [
                [rows[(suite, k, t)][field] for field in ("P_minus_B1_pp", "P_minus_reference_pp") for k in (1, 10)]
                for t in range(10)
            ]
        )
        assert np.abs(values).max() <= 40
        im = ax.pcolormesh(
            np.arange(5) - 0.5,
            np.arange(11) - 0.5,
            values,
            cmap=DIFFERENCE_CMAP,
            vmin=-40,
            vmax=40,
            shading="flat",
            edgecolors="#EEEEEE",
            linewidth=0.35,
            rasterized=False,
        )
        ax.set(
            xlim=(-0.5, 3.5),
            ylim=(9.5, -0.5),
            yticks=np.arange(10),
            yticklabels=[f"{t:02d} {TASK_LABELS[suite][t]}" for t in range(10)],
            xticks=np.arange(4),
            xticklabels=["K=1", "K=10", "K=1", "K=10"],
        )
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=3, labelsize=7.5)
        ax.text(0.5, 1.075, "P - B1", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.6)
        ax.text(2.5, 1.075, "P - released", transform=ax.get_xaxis_transform(), ha="center", fontsize=7.6)
        ax.text(-0.06, 1.20, f"({chr(97 + index)}) {SUITE_LABELS[index]}", transform=ax.transAxes, fontsize=9)
        ax.axvline(1.5, color="white", lw=2.5)
        ax.spines[:].set_visible(False)
        for row in range(10):
            for col in range(4):
                v = values[row, col]
                ax.text(
                    col,
                    row,
                    f"{v:+.1f}" if v else "0",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(v) >= 28 else INK,
                )
    cax = fig.add_axes((0.31, 0.055, 0.47, 0.014))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-40, -20, 0, 20, 40])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=7.5, width=0.5, length=2)
    cb.set_label("Success difference (pp)", fontsize=8)
    save(fig, "task_heatmap_all")


def literature_evidence(data):
    rows = data["literature_evidence"]
    assert all(r["status"] == "primary_verified" for r in rows)
    fig, axes = plt.subplots(2, 3, figsize=(7, 3.75))
    titles = ["Hermite-VLA", "Faithful Plans", "ESP", "DriftingVLA", "SAVLA", "AnyCamVLA"]
    protocols = [
        "LIBERO / pi0.5",
        "Custom Block-AR",
        "10-step vs. 1-step",
        "Full-model pi0.5",
        "10% GR00T data",
        "Novel camera views",
    ]
    for i, (ax, r) in enumerate(zip(axes.flat, rows, strict=True)):
        values = [r["baseline"], r["variant"]]
        labels = [r["baseline_label"], r["variant_label"]]
        if "intermediate" in r:
            values.insert(1, r["intermediate"])
            labels.insert(1, "View aug.")
        labels = [label.replace("conditioning", "cond.") for label in labels]
        positions = np.arange(len(values))
        ax.plot(positions, values, color="#777777", lw=0.9, zorder=2)
        ax.plot(positions, values, linestyle="", marker="o", color="#333333", ms=4, zorder=3)
        ax.plot(0, values[0], marker="o", mfc="white", mec="#333333", ms=4, zorder=4)
        for x, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:g}",
                (x, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        margin = 2.5 if max(values) - min(values) < 10 else 5
        ax.set(
            xlim=(-0.25, len(values) - 0.75),
            ylim=(min(values) - margin, min(100.8, max(values) + margin)),
            xticks=positions,
            xticklabels=labels,
            ylabel="Success (%)" if i % 3 == 0 else "",
        )
        ax.tick_params(axis="x", labelsize=7.5)
        panel(ax, chr(97 + i), titles[i])
        ax.text(0.5, 1.02, protocols[i], transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED)
        tidy(ax)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.89, bottom=0.13, wspace=0.4, hspace=0.75)
    save(fig, "literature_evidence")


def coverage():
    data = json.loads((HERE / "current-results.json").read_bytes())
    matrix = {(r["checkpoint_id"], r["num_steps"]): r for r in data["checkpoint_results"]}
    checkpoints = [f"{g}-seed{s}" for g in ("B0", "B1", "P") for s in SEEDS] + ["released-pi05-libero"]
    labels = [f"{g} / {s}" for g in ("B0", "B1", "P") for s in SEEDS] + ["Released"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6), sharey=True)
    for ax, k, letter in zip(axes, (1, 10), "ab", strict=True):
        selected = [matrix[(ck, k)]["observed"] for ck in checkpoints]
        accepted = [matrix[(ck, k)]["validated"] for ck in checkpoints]
        y = np.arange(10)
        ax.barh(y, selected, color="#D8D8D8", height=0.63, edgecolor="#777777", linewidth=0.3)
        ax.barh(y, accepted, color="#555555", height=0.63, linewidth=0)
        ax.set(xlim=(0, 2000), xticks=[0, 1000, 2000], yticks=y, yticklabels=labels, xlabel="Episodes")
        ax.set_ylim(9.65, -0.65)
        panel(ax, letter, f"K = {k}")
        tidy(ax, grid="x")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor="#555555", label="Original phase archives"),
            Patch(facecolor="#D8D8D8", edgecolor="#777777", linewidth=0.3, label="Other selected episodes"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 1.0),
        ncol=2,
        frameon=False,
        fontsize=7.8,
    )
    fig.subplots_adjust(left=0.12, right=0.965, top=0.8, bottom=0.18, wspace=0.12)
    save(fig, "coverage")


def draw_all(data):
    style()
    success_overview(data)
    contrast_intervals(data)
    task_differences(data)
    task_heatmap_all(data)
    literature_evidence(data)
    coverage()


if __name__ == "__main__":
    draw_all(json.loads((HERE / "figure-data.json").read_bytes()))
