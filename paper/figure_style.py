"""Shared publication dimensions, typography, and semantic colors."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WIDTH = 7.0
COLORS = {"B0": "#999999", "B1": "#D55E00", "P": "#0072B2", "reference": "#222222"}
INK = "#222222"
MUTED = "#666666"
GRID = "#D9D9D9"


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "font.weight": "normal",
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "axes.labelweight": "normal",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "scalar-pi-figures-v1",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": True,
        }
    )


def save_figure(fig, name):
    folder = HERE / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(
        folder / f"{name}.pdf",
        metadata={"Author": "Nima Kamali and Mert Turna", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(folder / f"{name}.png", dpi=300)
    svg = folder / f"{name}.svg"
    fig.savefig(svg, metadata={"Date": None, "Creator": "Scalar-Pi"})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf8").splitlines()) + "\n", encoding="utf8"
    )
    plt.close(fig)


style = setup_style
save = save_figure
