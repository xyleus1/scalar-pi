"""Shared publication dimensions, typography, and semantic colors."""

from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WIDTH = 7.0
COLORS = {"B0": "#999999", "B1": "#D55E00", "P": "#0072B2", "reference": "#222222"}
INK = "#222222"
MUTED = "#666666"
GRID = "#D9D9D9"
PALE = {"B0": "#F2F2F2", "B1": "#FBEDE5", "P": "#E6F1F7", "reference": "#EEEEEE"}
DIFFERENCE_CMAP = LinearSegmentedColormap.from_list("control_scalar", [COLORS["B1"], "white", COLORS["P"]])


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "font.weight": "normal",
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.5,
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
    fig.savefig(folder / f"{name}.svg", metadata={"Date": None, "Creator": "Scalar-Pi"})
    plt.close(fig)


def panel_label(ax, label, title="", *, x=0, y=1.04):
    ax.text(x, y, f"({label})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, weight="bold")
    if title:
        ax.text(x + 0.055, y, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=9)


style = setup_style
save = save_figure
