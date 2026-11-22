"""Regenerate the paper figures and tables from the retained descriptive results."""

import json
import math
from pathlib import Path

import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
mpl.use("Agg")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def tex(value):
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def design():
    fig, ax = plt.subplots(figsize=(10.5, 2.9))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#384b60", linewidth=1)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.12, 1.7, 2.1, 0.72, "Images + language\nFrozen VLM / cache", "#e8eef4")
    box(2.9, 1.35, 2.25, 1.0, "Original action expert\n" + r"$f_\theta(x,\tau,c)$", "#d8e8f0")
    box(5.85, 1.35, 1.7, 1.0, "Dot + sum\n" + r"$\Psi=x^\top f_\theta$", "#dcebdc")
    box(8.2, 1.35, 2.0, 1.0, "Full input gradient\n" + r"$v=\nabla_x\Psi$", "#f1e5cf")
    for a, b in [((2.3, 2.05), (2.8, 1.95)), ((5.25, 1.85), (5.75, 1.85)), ((7.65, 1.85), (8.1, 1.85))]:
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#384b60"})
    ax.text(1.1, 0.55, r"Noisy chunk $x$; time $\tau$", ha="center", fontsize=10)
    ax.annotate("", xy=(3.35, 1.28), xytext=(2.3, 0.7), arrowprops={"arrowstyle": "->", "lw": 1.4})
    ax.annotate(
        "",
        xy=(6.65, 1.26),
        xytext=(2.3, 0.6),
        arrowprops={"arrowstyle": "->", "lw": 1.4, "connectionstyle": "angle,angleA=0,angleB=90,rad=5"},
    )
    ax.text(8.9, 0.48, r"$f_\theta+J_{f_\theta}^{\top}x$", ha="center", fontsize=12)
    fig.savefig(HERE / "figures/method.pdf", bbox_inches="tight")
    fig.savefig(HERE / "figures/method.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    (HERE / "tables/design.tex").write_text(r"""\begin{table}[t]
\centering
\begin{tabular}{llll}
\toprule Arm & Trunk & Output initialization & Velocity \\
\midrule B0 & Pretrained & Pretrained & $f_\theta$ \\
B1 & Same pretrained trunk & Zero weight and bias & $f_\theta$ \\
P & Same initial tensors as B1 & Zero weight and bias & $\nabla_x(x^\top f_\theta)$ \\
\bottomrule
\end{tabular}
\caption{Matched controls. Each research arm uses seeds 17, 42, and 93, 30,000 updates, and the same frozen-backbone numerical protocol.}
\label{tab:design}
\end{table}
""")


def conclusion(points):
    if not math.isfinite(points):
        raise ValueError("Paired success difference must be finite")
    direction = "higher" if points > 0 else "lower"
    if points == 0:
        return "The scalar-potential expert and the reset direct-field control had the same success rate on matched ten-step rollouts (0.00 percentage points)."
    return (
        f"On matched ten-step LIBERO rollouts, the scalar-potential expert had a success rate "
        f"{abs(points):.2f} percentage points {direction} than the reset direct-field control. "
        "The descriptive 95\\% interval was [-1.27, +1.75] percentage points; unresolved hardware eligibility prevents a confirmatory benefit claim."
    )


def current_results():
    current = json.loads((HERE / "current-results.json").read_bytes())
    if current["kind"] != "scalar_pi_combined_observed_results_v1":
        raise ValueError("Unrecognized combined results")
    points = current["paired_difference_pp"]
    rows = current["checkpoint_results"]
    if sum(r["observed"] for r in rows) != current["distinct_episodes"] or current["matched_pairs"] <= 0:
        raise ValueError("Combined evidence disagrees with paper totals")
    matrix = {(r["checkpoint_id"], r["num_steps"]): r for r in rows}
    checkpoints = [f"{g}-seed{s}" for g in ("B0", "B1", "P") for s in (17, 42, 93)] + ["released-pi05-libero"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), sharey=True)
    for ax, k in zip(axes, (1, 10), strict=True):
        ax.barh(checkpoints, [2000] * len(checkpoints), color="#e7ebed", height=0.74, label="Planned remainder")
        observed = [matrix.get((ck, k), {}).get("observed", 0) for ck in checkpoints]
        validated = [matrix.get((ck, k), {}).get("validated", 0) for ck in checkpoints]
        ax.barh(checkpoints, observed, color="#cb701d", height=0.74, label="Other observed")
        ax.barh(checkpoints, validated, color="#244c70", height=0.74, label="Phase validated")
        ax.set(title=f"K = {k}", xlim=(0, 2000), xticks=(0, 1000, 2000))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    visible = {
        "Planned remainder": current["distinct_episodes"] < 40000,
        "Other observed": current["distinct_episodes"] > current["validated_episodes"],
        "Phase validated": current["validated_episodes"] > 0,
    }
    shown = [(handle, label) for handle, label in zip(handles[::-1], labels[::-1], strict=True) if visible[label]]
    if shown:
        fig.legend(
            [h for h, _ in shown], [label for _, label in shown], loc="lower center", ncol=len(shown), frameon=False
        )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(HERE / "figures/coverage.png", dpi=170)
    plt.close(fig)
    lines = [
        r"\paragraph{Observed outcomes.} At "
        + current["as_of_utc"]
        + ", the retained results contained "
        + f"{current['distinct_episodes']:,} distinct original episodes. Table~\\ref{{tab:observed}} "
        + "shows the success counts for every checkpoint and inference-step combination with reported outcomes.",
        r"\begin{table}[htbp]\centering\begin{tabular}{lrrr}\toprule",
        r"Checkpoint & Euler steps & Successful / observed & Phase validated \\ \midrule",
    ]
    lines.extend(
        tex(r["checkpoint_id"])
        + f" & {r['num_steps']} & {r['successes']:,} / {r['observed']:,} & {r['validated']:,} "
        + r"\\"
        for r in rows
    )
    lines.extend(
        [
            r"\bottomrule\end{tabular}\caption{Outcome counts across the observed hardware. "
            r"The final column counts episodes in the original accepted phase archives; task validation is reported separately. The planned count is "
            r"2,000 episodes per row.}\label{tab:observed}\end{table}",
            r"\paragraph{Paired comparison.} "
            f"The observed comparison uses {current['matched_pairs']:,} "
            "matched ten-step episodes per arm. " + current["support_sentence"],
            r"\begin{figure}[htbp]\centering\includegraphics[width=.87\linewidth]{figures/coverage.png}"
            r"\caption{Retained evaluation coverage for each checkpoint and inference-step count, "
            r"with 2,000 planned episodes per bar.}\end{figure}",
        ]
    )
    (HERE / "tables/results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "tables/conclusion.tex").write_text(conclusion(points) + "\n", encoding="utf-8")
    support = current["methods_sentence"].replace(
        "The registered interval is reported from the verified full-study analysis.",
        "The descriptive interval below reuses the original bootstrap; a hardware-qualified confirmatory interval is not asserted.",
    )
    (HERE / "tables/support.tex").write_text(support + "\n", encoding="utf-8")
    return r"\begin{center}\small Evaluation snapshot of " + current["as_of_utc"] + r".\end{center}"


if __name__ == "__main__":
    design()
    (HERE / "tables/status.tex").write_text(current_results() + "\n", encoding="utf8")
