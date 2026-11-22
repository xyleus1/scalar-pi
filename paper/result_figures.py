"""Publication figures from the retained study arrays, without new estimation."""

import json
from pathlib import Path

from figure_style import COLORS
from figure_style import GRID
from figure_style import INK
from figure_style import save
from figure_style import style
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SUITE_LABELS = ("Spatial", "Object", "Goal", "Long", "Overall")
GROUPS = ("B0", "B1", "P", "reference")
SEEDS = (17, 42, 93)
MARKERS = {"B0": "s", "B1": "^", "P": "o", "reference": "D"}
NAMES = {"B0": "B0: pretrained direct", "B1": "B1: reset direct", "P": "P: scalar", "reference": "Released"}
LONG_TASK_LABELS = [
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
]


def panel(ax, label, title):
    ax.set_title(f"({label}) {title}", loc="left", fontsize=9, fontweight="bold", pad=5)


def tidy(ax, *, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.spines[["left", "bottom"]].set_linewidth(0.5)
    ax.tick_params(length=2.5, width=0.5, pad=3, colors=INK, labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
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
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.55), sharex=True, sharey=True)
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
                        s=9,
                        marker=MARKERS[group],
                        color=COLORS[group],
                        edgecolor="white",
                        linewidth=0.35,
                        zorder=4,
                    )
                    ax.plot([x - 0.065, x + 0.065], [vals.mean()] * 2, color=COLORS[group], lw=1.5, zorder=3)
        panel(ax, letter, f"{k} Euler step" + ("s" if k != 1 else ""))
        ax.set(ylim=(70, 101.5), yticks=[70, 80, 90, 100], xlim=(-0.55, 4.55))
        ax.set(xticks=np.arange(5), xticklabels=SUITE_LABELS)
        ax.axvline(3.5, color="#BBBBBB", lw=0.55, ls=(0, (2, 2)))
        tidy(ax)
    axes[0].set_ylabel("Success (%)", fontsize=8)
    model_legend(fig, anchor=(0.54, 1.005))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.80, bottom=0.14, wspace=0.13)
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
    fig = plt.figure(figsize=(7, 2.65))
    ax = fig.add_axes((0.35, 0.20, 0.39, 0.69))
    label_ax = fig.add_axes((0.02, 0.20, 0.32, 0.69), sharey=ax)
    value_ax = fig.add_axes((0.765, 0.20, 0.225, 0.69), sharey=ax)
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
        label_ax.text(0, y, label, va="center", fontsize=8, color=INK, fontweight="bold" if primary else "normal")
        value_ax.text(
            1,
            y,
            f"{p:+.2f} [{lo:+.2f}, {hi:+.2f}]",
            ha="right",
            va="center",
            fontsize=8,
            color=INK,
        )
        ax.errorbar(
            p,
            y,
            xerr=[[p - lo], [hi - p]],
            fmt=MARKERS[left["group"]],
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
    label_ax.text(0, 8.05, "Comparison", fontsize=9, weight="bold")
    value_ax.text(1, 8.05, "Difference [95% interval]", ha="right", fontsize=9, weight="bold")
    save(fig, "contrast_intervals")


def task_differences(data):
    rows = {(r["num_steps"], r["task_id"]): r for r in data["task_comparisons"] if r["suite"] == "libero_10"}
    fig = plt.figure(figsize=(7, 2.9))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.25, 1], hspace=0.62, wspace=0.52, left=0.255, right=0.975, bottom=0.17, top=0.81
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
        yticklabels=[f"{t:02d}  {LONG_TASK_LABELS[t]}" for t in range(9, -1, -1)],
        ylim=(-0.65, 9.65),
        xlim=(-40, 40),
        xticks=[-40, -20, 0, 20, 40],
        xlabel="Success difference (pp)",
    )
    ax.tick_params(axis="y", labelsize=8, length=0)
    tidy(ax, grid="x")
    ax.spines["left"].set_visible(False)
    panel(ax, "a", "Long tasks, K = 10")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=COLORS["P"], ms=3, label="P - B1"),
            Line2D([], [], marker="D", ls="", color=COLORS["reference"], ms=3, label="P - released"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.41, 1.0),
        frameon=False,
        fontsize=8,
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
            fontsize=8,
            color=COLORS["reference"],
        )
        a.set_ylabel("Success (%)", labelpad=2, fontsize=8)
        if index == 0:
            a.tick_params(labelbottom=False)
        if index == 1:
            a.set_xlabel("Training seed", fontsize=8)
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
        fontsize=8,
        handletextpad=0.35,
        columnspacing=0.8,
    )
    save(fig, "task_differences")


def draw_all(data):
    style()
    success_overview(data)
    contrast_intervals(data)
    task_differences(data)


if __name__ == "__main__":
    draw_all(json.loads((HERE / "figure-data.json").read_bytes()))
