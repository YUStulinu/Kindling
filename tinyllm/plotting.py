"""Shared chart style: thin lines, a quiet grid, categorical colors in a fixed order."""

import matplotlib

matplotlib.use("Agg")  # render to file, no window
import matplotlib.pyplot as plt

# Colorblind-validated categorical palette; colors are assigned in order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0, "font.size": 9,
    })


def smooth(values, alpha=0.9):
    """Exponential moving average: the per-batch loss is noisy, the trend is what matters."""
    out, s = [], None
    for v in values:
        s = v if s is None else alpha * s + (1 - alpha) * v
        out.append(s)
    return out


def direct_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK_2)
    ax.plot([x], [y], "o", ms=4, color=color)
