"""Original computational schematics for the Scalar-Pi paper."""

import figure_style as fs
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def canvas(height):
    fig = plt.figure(figsize=(7, height))
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    ax.set(xlim=(0, 7), ylim=(0, height))
    ax.set_axis_off()
    return fig, ax


def label(ax, x, y, text, *, size=8, color=None, weight="normal", ha="center", va="center"):
    ax.text(x, y, text, fontsize=size, color=color or fs.INK, weight=weight, ha=ha, va=va, linespacing=1.18)


def arrow(ax, start, end, *, color=None, dashed=False, width=0.8):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "lw": width,
            "color": color or fs.INK,
            "linestyle": "--" if dashed else "-",
            "mutation_scale": 7,
            "shrinkA": 1,
            "shrinkB": 1,
        },
    )


def route(ax, points, *, color=None, dashed=False, width=0.8):
    col = color or fs.INK
    x, y = zip(*points[:-1], strict=True)
    ax.plot(x, y, color=col, lw=width, linestyle="--" if dashed else "-", solid_capstyle="butt")
    arrow(ax, points[-2], points[-1], color=col, dashed=dashed, width=width)


def module(ax, x, y, w, h, text="", *, color=None, fill="white", size=8, dashed=False):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fill, edgecolor=color or fs.INK, lw=0.75, linestyle="--" if dashed else "-")
    )
    if text:
        label(ax, x + w / 2, y + h / 2, text, size=size)


def tensor(ax, x, y, w, h, *, color=None, rows=4, cols=3, fill="white"):
    col = color or fs.INK
    module(ax, x, y, w, h, fill=fill, color=col)
    for j in range(1, cols):
        ax.plot([x + w * j / cols] * 2, [y, y + h], color=col, lw=0.35)
    for j in range(1, rows):
        ax.plot([x, x + w], [y + h * j / rows] * 2, color=col, lw=0.35)


def transformer(ax, x, y, w, h, text="Action expert", *, color=None):
    col = color or fs.INK
    for dx, dy in ((0.055, 0.055), (0.028, 0.028), (0, 0)):
        module(ax, x + dx, y + dy, w, h, fill="#F3F3F3", color=col)
    label(ax, x + w / 2, y + h / 2, text, size=8)


def operation(ax, x, y, text, *, color=None, r=0.13, size=8):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color or fs.INK, lw=0.8))
    label(ax, x, y, text, size=size, color=color)


def heading(ax, x, y, letter, text):
    label(ax, x, y, f"({letter})", ha="left", size=9, weight="bold")
    label(ax, x + 0.3, y, text, ha="left", size=9, weight="bold")


def method():
    fig, ax = canvas(2.9)
    blue, orange = fs.COLORS["P"], fs.COLORS["B1"]
    heading(ax, 0.08, 2.69, "a", "Shared action expert")
    heading(ax, 4.25, 2.69, "b", "Velocity readout")

    label(ax, 0.48, 2.22, "Images +\ninstruction", size=8)
    arrow(ax, (0.96, 2.22), (1.17, 2.22))
    module(ax, 1.2, 2.03, 1.15, 0.38, "Frozen VLM", fill="#EEEEEE", size=8)
    arrow(ax, (2.39, 2.22), (2.67, 2.22))
    label(ax, 3.02, 2.22, r"Cached $c$", size=8)
    route(ax, [(3.02, 2.04), (3.02, 1.62)])

    tensor(ax, 0.2, 1.04, 0.34, 0.5, rows=5, cols=4)
    label(ax, 0.37, 1.73, r"$x_\tau$", size=9)
    label(ax, 0.36, 0.89, r"$10\!\times\!32$", size=8)
    arrow(ax, (0.58, 1.29), (1.3, 1.29))
    transformer(ax, 1.34, 1.01, 2.19, 0.56, r"Action expert $f_\theta$")
    label(ax, 1.19, 1.82, r"$\tau$", size=9)
    route(ax, [(1.35, 1.82), (1.71, 1.82), (1.71, 1.63)])
    arrow(ax, (3.61, 1.29), (4.13, 1.29))

    route(ax, [(4.15, 1.29), (4.15, 2.08), (5.5, 2.08)], color=orange)
    label(ax, 4.71, 2.28, "Direct", size=8, color=orange)
    label(ax, 5.94, 2.08, r"$v_\theta=f_\theta$", size=10)
    arrow(ax, (4.15, 1.29), (4.65, 1.29), color=blue)
    operation(ax, 4.8, 1.29, r"$\cdot$", color=blue)
    label(ax, 4.79, 1.66, "Scalar", size=8, color=blue)
    route(ax, [(0.74, 1.29), (0.74, 0.81), (4.8, 0.81), (4.8, 1.15)], color=blue)
    arrow(ax, (4.95, 1.29), (5.19, 1.29), color=blue)
    label(ax, 5.37, 1.29, r"$\Psi_\theta$", size=10)
    arrow(ax, (5.57, 1.29), (5.81, 1.29), color=blue)
    label(ax, 6.38, 1.29, r"$v_\theta=\nabla_x\Psi_\theta$", size=10)
    label(ax, 6.04, 0.92, r"$=f_\theta+J_{f_\theta}^{\mathsf{T}}x_\tau$", size=9)
    route(ax, [(5.37, 1.09), (5.37, 0.6), (0.1, 0.6), (0.1, 1.04)], color=blue, dashed=True)
    label(ax, 2.75, 0.46, "Full input derivative through the action expert", size=8)

    ax.plot([0.1, 6.87], [0.31, 0.31], color=fs.GRID, lw=0.6)
    label(ax, 1.21, 0.15, r"3 arms $\times$ 3 training seeds", size=8)
    label(ax, 3.54, 0.15, r"$K=1$ or $10$ Euler steps", size=8)
    label(ax, 5.78, 0.15, "10 predicted / 5 executed actions", size=8)
    fs.save_figure(fig, "method")


def draw_all(data):
    del data
    fs.setup_style()
    method()
