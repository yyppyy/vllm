# =========================
# Reusable plotting helpers
# =========================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import cycler
from pathlib import Path

# 1) Color-blind-safe palettes you can reuse anywhere
PALETTES = {
    "okabe_ito": [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    ],
    "tableau10": [
        "#4E79A7", "#59A14F", "#B07AA1", "#F28E2B", "#E15759",
        "#76B7B2", "#EDC948", "#FF9DA7", "#9C755F", "#BAB0AC"
    ],
    "tol_bright": [
        "#4477AA", "#66CCEE", "#228833", "#CCBB44",
        "#EE6677", "#AA3377", "#BBBBBB", "#000000"
    ],
}

def get_palette(n: int, name: str = "okabe_ito"):
    base = PALETTES.get(name, PALETTES["okabe_ito"])
    if n <= len(base):
        return base[:n]
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def apply_color_cycle(n_series: int, name: str = "okabe_ito"):
    colors = get_palette(n_series, name)
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    return colors

def set_paper_style(*, base_font=11, dpi=300, grid_alpha=0.35):
    plt.rcParams.update({
        "figure.dpi": 180,
        # "savefig.dpi": dpi,
        # "savefig.bbox": "tight",
        # "savefig.pad_inches": 0.02,
        "font.size": base_font,
        "axes.titlesize": base_font + 1,
        "axes.labelsize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 2,
        "axes.titlepad": 8,
        "axes.labelpad": 6,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.6,
        "grid.alpha": grid_alpha,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

MARKERS = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h', 'p']
