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
    lines = [
        r"\paragraph{Observed outcomes.} The retained results contain "
        f"{current['distinct_episodes']:,} distinct original episodes. Table~\\ref{{tab:observed}} "
        "shows the success counts for every checkpoint and inference-step combination with reported outcomes.",
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
            r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/coverage.pdf}"
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
    return r"\begin{center}\small Study results, 22 November 2026.\end{center}"


if __name__ == "__main__":
    design()
    (HERE / "tables/status.tex").write_text(current_results() + "\n", encoding="utf8")
