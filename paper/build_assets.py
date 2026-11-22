"""Regenerate the paper tables from the retained descriptive results."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
    by_key = {(r["checkpoint_id"], r["num_steps"]): r for r in rows}
    checkpoints = list(dict.fromkeys(r["checkpoint_id"] for r in rows))
    require(len(by_key) == 20 and len(checkpoints) == 10, "Incomplete checkpoint grid")
    require(all(r["observed"] == 2000 for r in rows), "Unexpected episode denominator")
    lines = [
        r"\begin{table}[!htbp]\centering\small",
        r"\begin{tabular}{lrrrr}\toprule",
        r"& \multicolumn{2}{c}{$K=1$} & \multicolumn{2}{c}{$K=10$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Checkpoint & Successes & Archived & Successes & Archived \\ \midrule",
    ]
    for i, checkpoint in enumerate(checkpoints):
        if i in (3, 6, 9):
            lines.append(r"\addlinespace[3pt]")
        one, ten = (by_key[(checkpoint, k)] for k in (1, 10))
        name = "Released reference" if checkpoint == "released-pi05-libero" else checkpoint.replace("-seed", ", seed ")
        lines.append(
            tex(name)
            + f" & {one['successes']:,} & {one['validated']:,} & {ten['successes']:,} & {ten['validated']:,} "
            + r"\\"
        )
    lines.append(
        r"\bottomrule\end{tabular}\caption{Complete outcome counts. Each solver setting contains "
        r"2,000 selected episodes per checkpoint. Archived counts refer to the original accepted phase "
        r"archives (16,000 episodes overall), not to the 40,000 selected rows that passed task-level "
        r"validation. The cross-device qualification limitation applies to all mixed-device comparisons.}"
        r"\label{tab:observed}\end{table}"
    )
    (HERE / "tables/results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "tables/conclusion.tex").write_text(conclusion(points) + "\n", encoding="utf-8")
    support = current["methods_sentence"].replace(
        "The registered interval is reported from the verified full-study analysis.",
        "The descriptive interval below reuses the original bootstrap; a hardware-qualified confirmatory interval is not asserted.",
    )
    (HERE / "tables/support.tex").write_text(support + "\n", encoding="utf-8")
    return r"\begin{center}\small Study results, 22 November 2026.\end{center}"


if __name__ == "__main__":
    design()
    (HERE / "tables/status.tex").write_text(current_results() + "\n", encoding="utf8")
