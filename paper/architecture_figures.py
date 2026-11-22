"""Original computational schematics for the Scalar-Pi paper."""

import figure_style as fs
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np


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


def unet(ax, x, y, w, h):
    widths = (0.18, 0.22, 0.22, 0.18)
    heights = (0.85, 0.52, 0.52, 0.85)
    xs = (0, 0.25, 0.53, 0.82)
    for xx, ww, hh in zip(xs, widths, heights, strict=True):
        module(ax, x + xx * w, y + (1 - hh) * h / 2, ww * w, hh * h, fill="#F3F3F3")
    for j in range(3):
        arrow(ax, (x + (xs[j] + widths[j]) * w, y + h / 2), (x + xs[j + 1] * w, y + h / 2))
    ax.plot([x + 0.09 * w, x + 0.91 * w], [y + 0.95 * h] * 2, color=fs.MUTED, lw=0.5)


def operation(ax, x, y, text, *, color=None, r=0.13, size=8):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color or fs.INK, lw=0.8))
    label(ax, x, y, text, size=size, color=color)


def heading(ax, x, y, letter, text):
    label(ax, x, y, f"({letter})", ha="left", size=9, weight="bold")
    label(ax, x + 0.3, y, text, ha="left", size=9)


def comparison_map():
    fig, ax = canvas(3.12)
    blue, orange = fs.COLORS["P"], fs.COLORS["B1"]
    positions = ((0.08, 1.58), (3.69, 1.58), (0.08, 0.05), (3.69, 0.05))
    for (x, y), letter, title in zip(
        positions, "abcd", ("Direct action field", "Input-dot flow potential", "Scalar-Pi", "EnergyFlow"), strict=True
    ):
        heading(ax, x, y + 1.31, letter, title)

    x, y = positions[0]
    tensor(ax, x + 0.02, y + 0.45, 0.28, 0.43)
    label(ax, x + 0.16, y + 0.24, r"$x$")
    arrow(ax, (x + 0.32, y + 0.67), (x + 0.63, y + 0.67))
    transformer(ax, x + 0.65, y + 0.38, 1.05, 0.56, "Pretrained\ntransformer")
    arrow(ax, (x + 1.74, y + 0.67), (x + 2.12, y + 0.67), color=orange)
    tensor(ax, x + 2.15, y + 0.45, 0.28, 0.43, color=orange)
    label(ax, x + 2.28, y + 0.24, r"$v=f_\theta$", color=orange)
    label(ax, x + 1.45, y + 0.04, "OpenPI; unrestricted velocity", size=7.5, color=fs.MUTED)

    x, y = positions[1]
    label(ax, x + 0.05, y + 0.67, r"$x$")
    arrow(ax, (x + 0.14, y + 0.67), (x + 0.38, y + 0.67))
    unet(ax, x + 0.4, y + 0.42, 0.76, 0.5)
    label(ax, x + 0.78, y + 0.24, "U-Net", size=7.5)
    arrow(ax, (x + 1.18, y + 0.67), (x + 1.5, y + 0.67))
    operation(ax, x + 1.65, y + 0.67, r"$\cdot$")
    route(ax, [(x + 0.05, y + 0.53), (x + 0.05, y + 0.12), (x + 1.65, y + 0.12), (x + 1.65, y + 0.52)])
    arrow(ax, (x + 1.8, y + 0.67), (x + 2.08, y + 0.67))
    label(ax, x + 2.22, y + 0.67, r"$\Phi$")
    arrow(ax, (x + 2.37, y + 0.67), (x + 2.66, y + 0.67))
    label(ax, x + 2.92, y + 0.67, r"$\nabla_x\Phi$")
    label(ax, x + 1.55, y + 0.04, "Sun et al.; PDE fields", size=7.5, color=fs.MUTED)

    x, y = positions[2]
    label(ax, x + 0.05, y + 0.67, r"$x$")
    arrow(ax, (x + 0.14, y + 0.67), (x + 0.38, y + 0.67))
    transformer(ax, x + 0.4, y + 0.4, 0.93, 0.54, "Pretrained\ntransformer")
    arrow(ax, (x + 1.36, y + 0.67), (x + 1.64, y + 0.67), color=blue)
    operation(ax, x + 1.78, y + 0.67, r"$\cdot$", color=blue)
    route(ax, [(x + 0.05, y + 0.53), (x + 0.05, y + 0.15), (x + 1.78, y + 0.15), (x + 1.78, y + 0.52)], color=blue)
    arrow(ax, (x + 1.92, y + 0.67), (x + 2.15, y + 0.67), color=blue)
    label(ax, x + 2.27, y + 0.67, r"$\Psi$", color=blue)
    arrow(ax, (x + 2.41, y + 0.67), (x + 2.68, y + 0.67), color=blue)
    label(ax, x + 2.99, y + 0.67, r"$\nabla_x\Psi$", color=blue)
    label(ax, x + 1.55, y + 0.02, "Same transformer; full input derivative", size=7.5, color=fs.MUTED)

    x, y = positions[3]
    label(ax, x + 0.05, y + 0.67, r"$x$")
    arrow(ax, (x + 0.14, y + 0.67), (x + 0.38, y + 0.67))
    unet(ax, x + 0.4, y + 0.42, 0.68, 0.5)
    label(ax, x + 0.74, y + 0.24, "Temporal U-Net", size=7.5)
    arrow(ax, (x + 1.12, y + 0.67), (x + 1.4, y + 0.67))
    module(ax, x + 1.43, y + 0.43, 0.6, 0.48, "Pool\n+ MLP", size=7.5)
    arrow(ax, (x + 2.07, y + 0.67), (x + 2.28, y + 0.67))
    label(ax, x + 2.41, y + 0.67, r"$E_\theta$")
    arrow(ax, (x + 2.54, y + 0.67), (x + 2.72, y + 0.67))
    label(ax, x + 3.02, y + 0.67, r"$-\nabla_x E_\theta$", size=7.7)
    label(ax, x + 1.55, y + 0.02, "Scalar-head score model", size=7.5, color=fs.MUTED)
    fs.save_figure(fig, "comparison_map")


def method():
    fig, ax = canvas(3.95)
    ax.set_ylim(-0.3, 3.65)
    blue, orange = fs.COLORS["P"], fs.COLORS["B1"]
    heading(ax, 0.08, 3.46, "a", "One velocity evaluation")
    label(ax, 0.56, 2.99, "Images +\ninstruction", size=7.6)
    arrow(ax, (1.0, 2.99), (1.25, 2.99))
    module(ax, 1.27, 2.77, 1.25, 0.43, "Frozen VLM", fill="#EEEEEE")
    arrow(ax, (2.54, 2.99), (2.85, 2.99))
    label(ax, 3.02, 2.99, r"$c$")
    route(ax, [(3.02, 2.84), (3.02, 2.6), (2.8, 2.6), (2.8, 2.36)])
    label(ax, 3.85, 3.0, "Cache reused across solver steps", size=7.5, color=fs.MUTED, ha="left")
    tensor(ax, 0.3, 1.77, 0.37, 0.54, rows=5, cols=4)
    label(ax, 0.48, 2.49, r"$x_\tau$", size=9)
    label(ax, 0.48, 1.53, r"$10\!\times\!32$", size=7.5)
    arrow(ax, (0.71, 2.04), (1.04, 2.04))
    module(ax, 1.07, 1.84, 0.63, 0.4, "Input\nproj.", size=7.5)
    arrow(ax, (1.73, 2.04), (2.06, 2.04))
    transformer(ax, 2.1, 1.73, 1.39, 0.61, "Action transformer\n18 layers; width 1024")
    label(ax, 1.4, 2.53, r"$\tau$", size=9)
    route(ax, [(1.53, 2.53), (2.24, 2.53), (2.24, 2.36)])
    arrow(ax, (3.57, 2.04), (3.81, 2.04))
    module(ax, 3.84, 1.84, 0.6, 0.4, "Output\nproj.", size=7.5)
    arrow(ax, (4.47, 2.04), (4.7, 2.04), color=blue)
    tensor(ax, 4.73, 1.77, 0.34, 0.54, rows=5, cols=4, color=blue)
    label(ax, 4.9, 1.59, r"$f_\theta$", size=9)
    route(ax, [(4.9, 2.34), (4.9, 2.66), (5.91, 2.66)], color=orange)
    label(ax, 5.42, 2.83, "Direct control", size=7.5, color=orange)
    label(ax, 6.32, 2.66, r"$v=f_\theta$", size=9, color=orange)
    arrow(ax, (5.1, 2.04), (5.34, 2.04), color=blue)
    operation(ax, 5.5, 2.04, r"$\cdot$", color=blue)
    route(ax, [(0.88, 2.04), (0.88, 1.27), (5.5, 1.27), (5.5, 1.88)], color=blue)
    label(ax, 2.95, 1.42, "Inner product with the complete noisy chunk", size=7.5, color=blue)
    arrow(ax, (5.66, 2.04), (5.94, 2.04), color=blue)
    label(ax, 6.13, 2.04, r"$\Psi_\theta$", color=blue, size=10)
    route(ax, [(6.13, 1.85), (6.13, 0.99), (0.17, 0.99), (0.17, 1.77)], color=blue, dashed=True)
    label(
        ax,
        3.28,
        0.8,
        r"Full input derivative: $v_\theta=f_\theta+J_{f_\theta}^{\mathsf{T}}x_\tau$",
        color=blue,
        size=8.7,
    )

    heading(ax, 0.08, 0.43, "b", "Closed-loop sampling")
    label(ax, 0.72, 0.04, r"$x_1\sim\mathcal{N}(0,I)$", size=8)
    arrow(ax, (1.29, 0.04), (2.33, 0.04))
    label(ax, 1.81, 0.22, r"$K=1$ or $10$", size=7.5)
    for j in range(10):
        module(ax, 2.4 + 0.21 * j, -0.095, 0.19, 0.27, fill=blue if j < 5 else "white", color=blue)
    label(ax, 3.43, -0.23, "10 predicted actions", size=7.5)
    arrow(ax, (4.59, 0.04), (5.15, 0.04))
    label(ax, 5.98, 0.04, "Execute 5; observe and replan", size=7.5)
    fs.save_figure(fig, "method")


def study_design():
    fig, ax = canvas(2.8)
    heading(ax, 0.08, 2.62, "a", "Nine matched training runs")
    label(ax, 1.78, 2.29, "Training seed", size=7.5, color=fs.MUTED)
    xs = (1.23, 1.78, 2.33)
    for x, seed in zip(xs, (17, 42, 93), strict=True):
        label(ax, x, 2.07, str(seed), size=8)
    for y, arm, detail in ((1.72, "B0", "pretrained"), (1.28, "B1", "reset direct"), (0.84, "P", "reset scalar")):
        label(ax, 0.16, y, arm, color=fs.COLORS[arm], weight="bold", ha="left")
        label(ax, 0.66, y, detail, size=7.5)
        for x in xs:
            ax.scatter(x, y, s=31, color=fs.COLORS[arm], marker="s", zorder=3, linewidths=0)
    label(ax, 1.38, 0.49, "30,000 updates / run; same data + optimizer", size=7.5, color=fs.MUTED)
    ax.plot([0.11, 2.64], [0.29] * 2, color=fs.GRID, lw=0.7)
    ax.scatter(0.19, 0.08, marker="D", s=24, color=fs.COLORS["reference"])
    label(ax, 0.42, 0.08, "One released checkpoint; separate reference", size=7.5, ha="left")

    ax.plot([2.77, 2.77], [0.08, 2.5], color=fs.GRID, lw=0.65)
    arrow(ax, (2.87, 1.38), (3.15, 1.38))
    heading(ax, 3.26, 2.62, "b", "Fixed evaluation grid")
    label(ax, 3.34, 2.19, "10 checkpoints", ha="left", size=8)
    label(ax, 5.14, 2.19, r"$\times$", size=10)
    label(ax, 5.55, 2.19, "2 solver settings", ha="left", size=8)
    label(ax, 3.34, 1.91, "9 trained + 1 released", ha="left", size=7.5, color=fs.MUTED)
    label(ax, 5.58, 1.91, r"$K=1,10$", ha="left", size=7.5, color=fs.MUTED)
    rows = ((1.52, "Spatial"), (1.26, "Object"), (1.0, "Goal"), (0.74, "Long"))
    for y, name in rows:
        label(ax, 3.34, y, name, ha="left", size=7.5)
        ax.scatter(np.linspace(4.07, 5.71, 10), [y] * 10, s=10, color=fs.MUTED, marker="s", linewidths=0)
    label(ax, 6.36, 1.13, r"$\times\ 50$" + "\nstarts/task", size=8)
    label(ax, 4.89, 0.48, r"$4\ \mathrm{suites}\times10\ \mathrm{tasks}$", size=7.5, color=fs.MUTED)
    ax.plot([3.3, 6.86], [0.29] * 2, color=fs.INK, lw=0.65)
    label(ax, 5.06, 0.08, "40,000 episodes  /  800 task evaluations", size=8.5, weight="bold")
    fs.save_figure(fig, "study_design")


def future_studies():
    fig, ax = canvas(6.25)
    ax.set_ylim(-0.9, 5.35)
    blue = fs.COLORS["P"]
    heading(ax, 0.08, 5.16, "a", "Trajectory-shape supervision")
    heading(ax, 3.69, 5.16, "b", "Stage prediction")
    module(ax, 0.12, 4.55, 0.62, 0.33, r"$h_\theta$")
    arrow(ax, (0.77, 4.71), (1.02, 4.71))
    module(ax, 1.05, 4.5, 0.93, 0.44, "Hermite\nhead", color=blue, dashed=True)
    arrow(ax, (2.01, 4.71), (2.27, 4.71), color=blue)
    tensor(ax, 2.3, 4.49, 0.55, 0.44, rows=2, cols=4, color=blue)
    label(ax, 2.56, 4.3, r"$2\times4\times6$", size=7.5)
    t = np.linspace(0, 1, 100)
    y = 3 * t**2 - 2 * t**3
    ax.plot(0.25 + 1.15 * t, 3.78 + 0.32 * y, color=blue, lw=1.25)
    ax.plot(1.4 + 1.15 * t, 4.1 - 0.17 * y, color=blue, lw=1.25)
    ax.scatter([0.25, 1.4, 2.55], [3.78, 4.1, 3.93], s=13, color=blue)
    ax.plot([0.25, 2.55], [3.68] * 2, color=fs.GRID, lw=0.6)
    label(ax, 1.4, 3.5, "Two cubic segments; six arm controls", size=7.5)
    label(ax, 1.48, 3.29, r"$\mathcal{L}_{\rm flow}+10\mathcal{L}_{\rm recon}$", size=8.2)

    for j in range(4):
        module(ax, 3.78 + j * 0.21, 4.48 + j * 0.055, 0.16, 0.31, fill="#EEEEEE")
    label(ax, 4.14, 4.23, "State history", size=7.5)
    arrow(ax, (4.64, 4.71), (4.93, 4.71))
    module(ax, 4.97, 4.45, 1.07, 0.5, "Event + pose\n+ time-to-event", color=blue, size=7.5)
    label(ax, 6.55, 5.0, r"$c$", size=9)
    route(ax, [(6.4, 5.0), (5.5, 5.0), (5.5, 4.96)])
    route(ax, [(6.08, 4.7), (6.48, 4.7), (6.48, 3.91), (6.12, 3.91)], color=blue)
    transformer(ax, 4.88, 3.65, 1.2, 0.47, "Conditioned expert", color=blue)
    label(ax, 4.03, 3.91, r"$x,t,c$")
    arrow(ax, (4.41, 3.91), (4.83, 3.91))
    label(ax, 5.21, 3.3, "Condition on a predicted waypoint", size=7.5)

    ax.plot([0.08, 6.87], [3.05] * 2, color=fs.GRID, lw=0.6)
    heading(ax, 0.08, 2.81, "c", "Geometric invariance")
    heading(ax, 3.69, 2.81, "d", "Action-conditioned future change")
    label(ax, 0.61, 2.24, r"$(x,c)$", size=9)
    label(ax, 2.28, 2.24, r"$(Rx,T_Rc)$", size=9)
    arrow(ax, (1.02, 2.24), (1.76, 2.24))
    label(ax, 1.38, 2.45, r"$R$", size=8)
    arrow(ax, (0.61, 2.05), (0.61, 1.57), color=blue)
    arrow(ax, (2.28, 2.05), (2.28, 1.57), color=blue)
    label(ax, 0.61, 1.38, r"$\nabla_x\Psi$", color=blue, size=10)
    label(ax, 2.28, 1.38, r"$R\nabla_x\Psi$", color=blue, size=10)
    arrow(ax, (1.06, 1.38), (1.79, 1.38), color=blue)
    label(ax, 1.43, 1.62, r"$R$", size=8, color=blue)
    label(ax, 1.44, 0.99, r"$\Psi(Rx,t,T_Rc)=\Psi(x,t,c)$", size=8)
    label(ax, 1.44, 0.77, "Requires the full potential to be invariant", size=7.5)

    tensor(ax, 3.85, 1.83, 0.31, 0.44, rows=3, cols=3)
    label(ax, 4.01, 1.6, r"$h_t$", size=8)
    arrow(ax, (4.19, 2.05), (4.54, 2.05))
    module(ax, 4.58, 1.82, 1.03, 0.46, "Future predictor", color=blue, dashed=True, size=7.5)
    for j in range(5):
        module(ax, 4.59 + j * 0.2, 2.47, 0.17, 0.15, color=blue, fill="white")
    label(ax, 4.15, 2.54, r"$A_{t:t+4}$", size=8)
    arrow(ax, (5.09, 2.45), (5.09, 2.3), color=blue)
    arrow(ax, (5.65, 2.05), (5.95, 2.05), color=blue)
    tensor(ax, 6.0, 1.83, 0.31, 0.44, rows=3, cols=3, color=blue)
    label(ax, 6.16, 1.6, r"$\widehat{\Delta e}$", size=8, color=blue)
    label(ax, 5.06, 1.16, r"Target: $\mathrm{sg}(e_{t+5}-e_t)$", size=8)
    label(ax, 5.06, 0.77, "Five demonstrated actions; frozen encoder target", size=7.5)

    ax.plot([0.08, 6.87], [0.54] * 2, color=fs.GRID, lw=0.6)
    heading(ax, 0.08, 0.27, "e", "Convex distributional generator")
    label(ax, 0.48, -0.2, r"$z\sim\mathcal{N}(0,I)$", size=8)
    arrow(ax, (1.17, -0.2), (1.56, -0.2))
    module(ax, 1.59, -0.43, 1.1, 0.46, r"$\Phi_\theta(z,c,G)$", color=blue, size=9)
    label(ax, 2.14, -0.61, r"Convex in $z$", size=7.5)
    arrow(ax, (2.73, -0.2), (3.08, -0.2), color=blue)
    label(ax, 3.4, -0.2, r"$\nabla_z\Phi_\theta$", color=blue, size=10)
    arrow(ax, (3.79, -0.2), (4.13, -0.2), color=blue)
    tensor(ax, 4.16, -0.42, 0.44, 0.45, rows=5, cols=3, color=blue)
    label(ax, 4.38, -0.6, r"$\widehat{A}\in\mathbb{R}^{60}$", color=blue, size=7.5)
    arrow(ax, (4.64, -0.2), (5.03, -0.2), color=blue)
    module(ax, 5.07, -0.43, 1.59, 0.46, "Energy-score loss", dashed=True, size=8)
    label(ax, 5.89, -0.61, r"Independent $\widehat{A},\widehat{A}'$; target $A$", size=7.5)
    fs.save_figure(fig, "future_studies")


def draw_all(data):
    del data
    fs.setup_style()
    comparison_map()
    method()
    study_design()
    future_studies()
