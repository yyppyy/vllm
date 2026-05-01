# SPDX-License-Identifier: Apache-2.0
"""Unified plotting style for figures targeted at top-tier
computer-systems venues (OSDI / SOSP / NSDI / ATC / ASPLOS / EuroSys /
SIGMOD).

Conventions:
  * Type 42 (TrueType) fonts in PDF/PS so editors can edit text.
  * Sans-serif (Helvetica/Arial fallback to DejaVu Sans) at 8-9 pt.
  * Subtle grids; thin spines; ticks pointing inward.
  * Color-blind-safe Okabe-Ito palette by default; black-and-white
    legibility via per-series markers, line styles, and hatches.

Top-level API:
  * apply_style(...) -- once at the top of a script; sets rcParams.
  * paper_figure(width=..., ratio=..., n_axes=...) -> (fig, axes)
        Sized for one or both columns of a two-column paper.
  * save_fig(fig, path) -- saves PDF + (optionally) PNG with sensible
        bbox / pad / dpi for camera-ready submission.
  * palette(n, name=...) -- returns a list of colors of length n.
  * line_styles(n), markers(n), hatches(n) -- companion enumerators.
  * style_axes(ax, ...), style_legend(ax, ...) -- post-hoc tidy-ups.

Plot helpers (thin wrappers on top of matplotlib that apply our
preferred defaults):
  * line(ax, x, y, *, label, series, ...)
  * scatter(ax, x, y, *, label, series, ...)
  * bar(ax, x, y, *, series, ...)
  * box(ax, data, positions=None, *, label=None, ...)

These helpers all take an integer `series` index (or a string label)
and look up a consistent color/marker/hatch/linestyle for that series.
Pass `series=0` for the first series, `series=1` for the second, etc.,
or pass an explicit `color=` to override.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cycler


# ---------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------

PALETTES = {
    # Wong (2011). Color-blind-safe; the de facto choice for
    # systems papers in the last few years.
    "okabe_ito": [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#56B4E9", "#CC79A7", "#F0E442", "#000000",
    ],
    # Tableau 10 (medium). Solid for >6 series.
    "tableau10": [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ],
    # Paul Tol's bright palette. Very high-contrast.
    "tol_bright": [
        "#4477AA", "#EE6677", "#228833", "#CCBB44",
        "#66CCEE", "#AA3377", "#BBBBBB", "#000000",
    ],
}
DEFAULT_PALETTE = "okabe_ito"

# ---------------------------------------------------------------------
# METRO-vs-others palette
# ---------------------------------------------------------------------
#
# When a figure compares METRO to baselines, METRO is the single warm
# series (vermilion); every other system uses a cold blue/green/slate
# from a curated cold family. The contrast is deliberate: warm/cold
# pre-attentively separates "our system" from "baselines" without the
# reader having to consult the legend.
#
# The colors are color-blind-safe (Wong / Okabe-Ito + Tol) and stay
# legible in B&W reprints when paired with distinct markers/hatches.

# Bright, paper-friendly palette. The cold colors are pushed to
# higher saturation than the standard Okabe-Ito set so they remain
# readable in print without losing their cool/cold identity. METRO
# stays a saturated red (~ Tableau "red" / matplotlib default).
METRO_WARM = "#E03131"   # bright red (Open Color "red 8")
COLD_FAMILY = [
    "#1F77B4",  # bright blue (matplotlib C0)
    "#2CA02C",  # bright green (matplotlib C2)
    "#17BECF",  # bright cyan (matplotlib C9) — clearly non-green
    "#3498DB",  # bright sky blue (Flat UI "peter river")
    "#5D6D7E",  # slate (last-resort neutral)
]


def metro_palette(n_others: int) -> tuple[str, list[str]]:
    """Return ``(metro_color, [cold_color, ...])`` for system
    comparisons. ``n_others`` is the number of non-METRO series.
    Cold colors cycle through ``COLD_FAMILY`` if more than five are
    needed, but in practice 1-4 is the realistic range.
    """
    cold = [COLD_FAMILY[i % len(COLD_FAMILY)] for i in range(n_others)]
    return METRO_WARM, cold


def palette(n: int, name: str = DEFAULT_PALETTE) -> list[str]:
    """Return a list of `n` colors from the named palette, repeating
    if necessary."""
    base = PALETTES.get(name, PALETTES[DEFAULT_PALETTE])
    if n <= len(base):
        return list(base[:n])
    reps = -(-n // len(base))  # ceil divide
    return list((base * reps)[:n])


# ---------------------------------------------------------------------
# Per-series enumerators (designed to remain legible in B&W reprints)
# ---------------------------------------------------------------------

MARKERS = ['o', 's', '^', 'D', 'v', 'X', 'P', '*', 'h', 'p']
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)),
               (0, (5, 2)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1))]
HATCHES = ['', '///', '\\\\\\', 'xxx', '...', '+++', 'ooo', '***',
           '|||', '---']


def markers(n: int) -> list[str]:
    return [MARKERS[i % len(MARKERS)] for i in range(n)]


def line_styles(n: int) -> list:
    return [LINE_STYLES[i % len(LINE_STYLES)] for i in range(n)]


def hatches(n: int) -> list[str]:
    return [HATCHES[i % len(HATCHES)] for i in range(n)]


# ---------------------------------------------------------------------
# Style application
# ---------------------------------------------------------------------

def apply_style(*, base_font: float = 11.0,
                palette_name: str = DEFAULT_PALETTE,
                use_tex: bool = False,
                font_family: str = "sans-serif") -> None:
    """Apply the unified style to matplotlib's rcParams. Call once at
    the top of a plotting script.

    base_font   point size of axis labels / ticks / legend (8-9 typical
                for two-column figures).
    palette_name name in PALETTES; sets the default color cycle.
    use_tex     if True, render text with LaTeX (slow). Off by default
                so plots build on machines without a TeX install.
    font_family one of "sans-serif" (Helvetica-like) or "serif"
                (Computer Modern / Times-like). Sans-serif is the
                modern systems-paper convention.
    """
    sans_stack = ['Helvetica', 'Arial', 'Liberation Sans',
                  'DejaVu Sans']
    serif_stack = ['Times New Roman', 'Times', 'Liberation Serif',
                   'DejaVu Serif']
    rc = {
        # PDF embedding -- editors can re-edit text in the figure.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",

        "text.usetex": bool(use_tex),
        "font.family": font_family,
        "font.sans-serif": sans_stack,
        "font.serif": serif_stack,
        "mathtext.fontset": "stix",

        # Sizes
        "font.size":         base_font,
        "axes.titlesize":    base_font,
        "axes.labelsize":    base_font,
        "xtick.labelsize":   base_font - 0.5,
        "ytick.labelsize":   base_font - 0.5,
        "legend.fontsize":   base_font - 1.0,
        "figure.titlesize":  base_font + 1,

        # Layout
        "axes.titlepad": 4,
        "axes.labelpad": 3,
        "axes.linewidth": 0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "axes.axisbelow": True,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,

        # Grid
        "grid.linewidth": 0.4,
        "grid.linestyle": "--",
        "grid.color": "#B0B0B0",
        "grid.alpha": 0.5,

        # Lines / markers / patches
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.8,
        "patch.linewidth": 0.6,
        "patch.edgecolor": "black",

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.borderaxespad": 0.4,

        # Figures / save
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # IMPORTANT: keep savefig.bbox unset (None) so the saved PDF
        # uses the figure's declared figsize verbatim. Setting this to
        # "tight" lets label content (which varies per figure) drive
        # the saved size and breaks the "all figures same size" rule.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,

        # Color cycle
        "axes.prop_cycle": cycler(color=palette(8, palette_name)),
    }
    plt.rcParams.update(rc)


# ---------------------------------------------------------------------
# Figure sizing — UNIFIED for this project
# ---------------------------------------------------------------------
#
# All figures are emitted at a standard physical size derived from the
# "four panels per row" layout of a two-column paper. Each panel is
# 1/4 of the 7-inch text width = 1.75" wide, with a 16:9 aspect ratio.
# When the resulting PDF is included into the paper at its native size
# the in-figure font (set by `apply_style(base_font=11)`) renders at
# the same physical 11 pt as 11 pt body text.
#
# Multi-panel figures scale the *width* (and height) with the panel
# grid so each panel keeps the standard 1.75 x 0.984 footprint and the
# 16:9 aspect ratio.

# Source-PDF panel size. Calibrated against the reference figure
# `plots/activated_experts_g8_ep8_bs32_likaixin_InstructCoder.pdf`,
# which is 3.208 in wide with 11 pt embedded fonts and looks correct
# when included in the paper four-per-row. The font/figure-width ratio
# of that reference is 11 / (3.208 * 72) ≈ 4.8 %, which we match by
# emitting source figures at 3.2 in wide with `base_font=11`. When the
# user `\includegraphics`-scales four of these into a 7-inch row they
# land at 1.75 in each on paper, with an on-paper font size of about
# 6 pt (the visual ratio is preserved by the scaling).
STANDARD_PANEL_WIDTH = 3.2    # inches
STANDARD_ASPECT = 3 / 4       # height / width
STANDARD_PANEL_HEIGHT = STANDARD_PANEL_WIDTH * STANDARD_ASPECT  # 2.4"

# Kept for backwards compatibility; these are no longer how scripts
# size their figures, but a few legacy callers still reference them.
WIDTH_PRESETS = {
    "single":         STANDARD_PANEL_WIDTH,
    "single_acm":     STANDARD_PANEL_WIDTH,
    "single_usenix":  STANDARD_PANEL_WIDTH,
    "double":         STANDARD_PANEL_WIDTH,
    "double_acm":     STANDARD_PANEL_WIDTH,
    "double_usenix":  STANDARD_PANEL_WIDTH,
    "wide":           STANDARD_PANEL_WIDTH,
    "third":          STANDARD_PANEL_WIDTH,
    "half":           STANDARD_PANEL_WIDTH,
    "quarter":        STANDARD_PANEL_WIDTH,
}

GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # kept for callers; unused here


def figure_width(width) -> float:
    """Backwards-compat shim. The standard width is enforced by
    `paper_figure`; this helper now ignores its argument."""
    return STANDARD_PANEL_WIDTH


# Per-panel margins, expressed as fractions of the panel size.
# Tuned for a 3.2 x 2.4 in panel with 11 pt labels:
#   left   ~ 0.55 in  (rotated y-label 0.20 + tick labels 0.30 + pad)
#   bottom ~ 0.50 in  (x-label 0.20 + tick labels 0.20 + pad)
PANEL_MARGINS = dict(left=0.17, right=0.98, bottom=0.21, top=0.97)


def paper_figure(width=None,
                 ratio: float = STANDARD_ASPECT,
                 height: Optional[float] = None,
                 n_axes: int | tuple[int, int] = 1,
                 sharex: bool = False,
                 sharey: bool = False,
                 wspace: float = 0.30,
                 hspace: float = 0.40,
                 **subplot_kw):
    """Create a figure at the project-standard size.

    Every panel is STANDARD_PANEL_WIDTH × STANDARD_PANEL_HEIGHT. Multi-
    panel figures scale the *grid*: `n_axes=N` returns one row of N
    panels; `n_axes=(rows, cols)` returns a (rows × cols) grid.

    Subplot margins are pinned via `subplots_adjust` so the data axes
    occupy the same fraction of every figure regardless of label
    content. **`save_fig` saves at the declared figsize** (no
    bbox=tight, no tight_layout) so figures across the paper are
    bit-for-bit the same physical size.

    `width`, `ratio`, and `height` are accepted for backwards
    compatibility but ignored.
    """
    if isinstance(n_axes, tuple):
        rows, cols = n_axes
    else:
        rows, cols = 1, int(n_axes)
    fig_w = STANDARD_PANEL_WIDTH * cols
    fig_h = STANDARD_PANEL_HEIGHT * rows
    fig, axes = plt.subplots(rows, cols,
                              figsize=(fig_w, fig_h),
                              sharex=sharex,
                              sharey=sharey,
                              **subplot_kw)
    # Convert the per-panel margin fractions into figure-space margins.
    # Outer margins are scaled by 1/cols (left/right) and 1/rows
    # (bottom/top) so that on a multi-panel figure the visible data
    # area per panel is the same as on a single-panel figure.
    fig.subplots_adjust(
        left=PANEL_MARGINS["left"] / cols,
        right=1.0 - (1.0 - PANEL_MARGINS["right"]) / cols,
        bottom=PANEL_MARGINS["bottom"] / rows,
        top=1.0 - (1.0 - PANEL_MARGINS["top"]) / rows,
        wspace=wspace,
        hspace=hspace,
    )
    return fig, axes


# ---------------------------------------------------------------------
# Axes / legend tidy-ups
# ---------------------------------------------------------------------

# Axis-scale prefixes — pick the largest divisor whose label has fewer
# digits than the raw value. Standard SI / engineering names.
_SI_SCALES = [
    (1e9, 'G'),
    (1e6, 'M'),
    (1e3, 'K'),
]

# Default cap on the number of major tick labels per axis.
DEFAULT_MAX_TICKS = 8


def _pick_axis_scale(values: Iterable[float],
                     threshold: float = 1e3) -> tuple[float, str]:
    """Pick (divisor, prefix) so the largest |value| sits in ~[0, 1000)
    after dividing. Returns (1.0, '') when no scaling needed."""
    vals = [v for v in values
            if v is not None and not (isinstance(v, float)
                                      and (v != v or
                                           v in (float('inf'),
                                                  float('-inf'))))]
    if not vals:
        return 1.0, ''
    max_abs = max(abs(float(v)) for v in vals)
    if max_abs < threshold:
        return 1.0, ''
    for divisor, prefix in _SI_SCALES:
        if max_abs >= divisor:
            return float(divisor), prefix
    return 1.0, ''


def _insert_axis_prefix(label: str, prefix: str) -> str:
    """Insert an SI prefix into a unit-bearing label.

    'Throughput (tok/s)' + 'K' -> 'Throughput (K tok/s)'
    'Latency'            + 'K' -> 'Latency (×10³)'
    """
    if not prefix:
        return label
    import re as _re
    m = _re.search(r'\(([^)]+)\)', label)
    if m:
        unit = m.group(1).strip()
        return label[:m.start()] + f'({prefix} {unit})' + label[m.end():]
    exp = {'K': 3, 'M': 6, 'G': 9}.get(prefix, 0)
    if exp:
        return f'{label} (×10$^{{{exp}}}$)'
    return label


def _apply_axis_scale_formatter(axis, divisor: float) -> None:
    """Display tick values divided by `divisor` (no data mutation)."""
    from matplotlib.ticker import FuncFormatter
    if divisor <= 1.0:
        return

    def _fmt(v, _pos):
        scaled = v / divisor
        if abs(scaled) < 1e-9:
            return '0'
        # Prefer integer rendering when divisor cleanly absorbs.
        if abs(scaled - round(scaled)) < 1e-6:
            return f'{int(round(scaled))}'
        return f'{scaled:g}'

    axis.set_major_formatter(FuncFormatter(_fmt))


def _cap_axis_ticks(axis, max_ticks: int = DEFAULT_MAX_TICKS) -> None:
    """Ensure `axis` shows at most `max_ticks` major labels.

    For numeric axes (AutoLocator/MaxNLocator) the locator is replaced
    by MaxNLocator(nbins=max_ticks). For categorical axes set up via
    `set_xticks` (FixedLocator), tick *positions* are kept (so grid
    lines and bar/box widths line up) but labels are blanked out
    everywhere except every k-th tick.
    """
    from matplotlib.ticker import (FixedLocator, MaxNLocator,
                                    AutoLocator, LogLocator)
    locator = axis.get_major_locator()
    ticks = axis.get_majorticklocs()
    n = len(ticks)
    if n <= max_ticks or max_ticks <= 0:
        return
    if isinstance(locator, FixedLocator):
        labels = [t.get_text() for t in axis.get_majorticklabels()]
        if not labels or all(not l for l in labels):
            return
        step = -(-n // max_ticks)  # ceil-divide
        # Strict striding: keep every step-th label and blank the
        # rest. We *don't* force the trailing label to stay visible
        # — that breaks the uniform spacing and crowds the end of the
        # axis when the trailing tick happens to fall mid-stride.
        new_labels = [lbl if i % step == 0 else ''
                       for i, lbl in enumerate(labels)]
        axis.set_ticklabels(new_labels)
        return
    if isinstance(locator, (AutoLocator, MaxNLocator)) or \
            type(locator).__name__ == 'AutoLocator':
        axis.set_major_locator(MaxNLocator(nbins=max_ticks))
        return
    # LogLocator and exotic locators: leave alone.
    if isinstance(locator, LogLocator):
        return


def style_axes(ax,
               *,
               x_label: Optional[str] = None,
               y_label: Optional[str] = None,
               x_lim: Optional[tuple] = None,
               y_lim: Optional[tuple] = None,
               y_zero: bool = False,
               x_log: bool = False,
               y_log: bool = False,
               grid_axis: str = "both",
               minor_ticks: bool = False,
               int_x: bool = False,
               int_y: bool = False,
               max_ticks: int = DEFAULT_MAX_TICKS,
               auto_scale_y: bool = True,
               auto_scale_x: bool = False) -> None:
    """Convenience: set common axis attributes in one call.

    Beyond the obvious lim/label arguments, this helper enforces three
    publication-style defaults that previously had to be repeated at
    every call site:

      * `auto_scale_y=True`  -> when the y range exceeds 1e3 / 1e6 / 1e9
        the tick labels are divided by the appropriate power of ten and
        the y-label gains a `K`, `M`, or `G` prefix
        (`tok/s` -> `K tok/s`).
      * `grid_axis='both'`  -> grid on both axes by default.
      * `max_ticks=8`       -> never more than 8 visible tick labels per
        axis. For categorical (FixedLocator) axes labels are thinned
        without disturbing the underlying tick positions, so bar/box
        widths and grid lines stay aligned.
    """
    if x_label is not None:
        ax.set_xlabel(x_label)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if y_zero:
        bot, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=max(top, bot))
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    # Auto-scale numeric axes (run before the locator cap so the
    # resulting tick set is what we re-format).
    if auto_scale_y and not y_log:
        ymin, ymax = ax.get_ylim()
        divisor, prefix = _pick_axis_scale([ymin, ymax])
        if divisor > 1.0:
            _apply_axis_scale_formatter(ax.yaxis, divisor)
            if y_label is not None:
                y_label = _insert_axis_prefix(y_label, prefix)
    if auto_scale_x and not x_log:
        xmin, xmax = ax.get_xlim()
        divisor, prefix = _pick_axis_scale([xmin, xmax])
        if divisor > 1.0:
            _apply_axis_scale_formatter(ax.xaxis, divisor)
            if x_label is not None:
                ax.set_xlabel(_insert_axis_prefix(x_label, prefix))

    if y_label is not None:
        ax.set_ylabel(y_label)

    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis,
                 linewidth=plt.rcParams["grid.linewidth"],
                 linestyle=plt.rcParams["grid.linestyle"],
                 color=plt.rcParams["grid.color"],
                 alpha=plt.rcParams["grid.alpha"])
    elif grid_axis == "off":
        ax.grid(False)
    if minor_ticks:
        ax.minorticks_on()
    if int_x:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if int_y:
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Cap number of tick labels.
    if max_ticks > 0:
        _cap_axis_ticks(ax.xaxis, max_ticks)
        _cap_axis_ticks(ax.yaxis, max_ticks)


def style_legend(ax, *,
                 loc: str = "best",
                 ncol: int = 1,
                 title: Optional[str] = None,
                 outside: bool = False,
                 **kwargs) -> Optional[mpl.legend.Legend]:
    """Wrapper around ax.legend() that applies our preferred defaults."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    kw = dict(loc=loc, ncol=ncol, title=title)
    if outside:
        kw["loc"] = "upper center"
        kw["bbox_to_anchor"] = (0.5, 1.18)
        kw["frameon"] = False
    kw.update(kwargs)
    return ax.legend(handles, labels, **kw)


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------

def save_fig(fig, path, *, also_png: bool = False,
             tight: bool = False, pad_inches: float = 0.02) -> Path:
    """Save a figure as PDF (default) and optionally PNG.

    The canvas is preserved at the declared `figsize` — neither
    `tight_layout()` nor `bbox_inches="tight"` is applied, because
    both can mutate the saved size in label-content-dependent ways
    and that breaks the project rule that all figures be the same
    physical size. Margins are pinned by `paper_figure(...)` instead.

    Returns the PDF path. Creates parent dirs as needed.

    `tight=True` re-enables the legacy auto-shrink behavior (only for
    rare callers that need the saved canvas to wrap their content
    snugly; most should leave it off).
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(p, bbox_inches="tight", pad_inches=pad_inches)
    else:
        # Explicit bbox_inches=None forces matplotlib to write the
        # canvas at its declared figsize regardless of any rcParams
        # leftover.
        fig.savefig(p, bbox_inches=None)
    if also_png:
        png = p.with_suffix(".png")
        if tight:
            fig.savefig(png, bbox_inches="tight",
                        pad_inches=pad_inches,
                        dpi=plt.rcParams["savefig.dpi"])
        else:
            fig.savefig(png, bbox_inches=None,
                        dpi=plt.rcParams["savefig.dpi"])
    return p


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _series_color(series, palette_name: str = DEFAULT_PALETTE):
    if isinstance(series, str):
        # Hash a string label into a stable index.
        idx = (hash(series) & 0x7fffffff) % len(PALETTES[palette_name])
    else:
        idx = int(series)
    return palette(idx + 1, palette_name)[idx]


def _series_marker(series):
    if isinstance(series, str):
        idx = (hash(series) & 0x7fffffff) % len(MARKERS)
    else:
        idx = int(series)
    return MARKERS[idx % len(MARKERS)]


def _series_linestyle(series):
    if isinstance(series, str):
        idx = (hash(series) & 0x7fffffff) % len(LINE_STYLES)
    else:
        idx = int(series)
    return LINE_STYLES[idx % len(LINE_STYLES)]


def _series_hatch(series):
    if isinstance(series, str):
        idx = (hash(series) & 0x7fffffff) % len(HATCHES)
    else:
        idx = int(series)
    return HATCHES[idx % len(HATCHES)]


def line(ax, x, y, *,
         label: Optional[str] = None,
         series: int | str = 0,
         marker: bool = True,
         color: Optional[str] = None,
         linestyle: Optional = None,
         palette_name: str = DEFAULT_PALETTE,
         **kwargs):
    """Line plot with our defaults (per-series color + linestyle +
    marker for B&W legibility)."""
    c = color or _series_color(series, palette_name)
    ls = linestyle or _series_linestyle(series)
    mk = _series_marker(series) if marker else None
    return ax.plot(x, y, color=c, linestyle=ls, marker=mk,
                   label=label, **kwargs)


def scatter(ax, x, y, *,
            label: Optional[str] = None,
            series: int | str = 0,
            color: Optional[str] = None,
            marker: Optional[str] = None,
            palette_name: str = DEFAULT_PALETTE,
            **kwargs):
    c = color or _series_color(series, palette_name)
    mk = marker or _series_marker(series)
    return ax.scatter(x, y, c=c, marker=mk, label=label, **kwargs)


def bar(ax, x, y, *,
        label: Optional[str] = None,
        series: int | str = 0,
        color: Optional[str] = None,
        hatch: Optional[str] = None,
        edgecolor: str = "black",
        palette_name: str = DEFAULT_PALETTE,
        **kwargs):
    """Bar plot with hatched fills for B&W legibility."""
    c = color or _series_color(series, palette_name)
    h = hatch if hatch is not None else _series_hatch(series)
    return ax.bar(x, y, color=c, hatch=h, edgecolor=edgecolor,
                  label=label, **kwargs)


def box(ax, data, *,
        positions: Optional[Sequence[float]] = None,
        widths: float | Sequence[float] = 0.5,
        showfliers: bool = False,
        median_color: str = "#D55E00",
        face_color: Optional[str] = None,
        face_alpha: float = 0.0,
        manage_ticks: bool = True,
        **kwargs):
    """Box plot with thin spines and a colored median line."""
    if positions is None:
        positions = list(range(1, len(data) + 1))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=widths,
        showfliers=showfliers,
        manage_ticks=manage_ticks,
        patch_artist=face_color is not None or face_alpha > 0,
        medianprops=dict(color=median_color, linewidth=1.2),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.7),
        capprops=dict(linewidth=0.7),
        flierprops=dict(marker='o', markersize=2, alpha=0.4),
        **kwargs,
    )
    if face_color is not None or face_alpha > 0:
        for patch in bp.get('boxes', []):
            patch.set_facecolor(face_color or "#0072B2")
            patch.set_alpha(face_alpha if face_alpha > 0 else 0.4)
    return bp


__all__ = [
    "PALETTES", "DEFAULT_PALETTE",
    "METRO_WARM", "COLD_FAMILY", "metro_palette",
    "MARKERS", "LINE_STYLES", "HATCHES",
    "WIDTH_PRESETS", "GOLDEN_RATIO",
    "palette", "markers", "line_styles", "hatches",
    "apply_style", "paper_figure", "save_fig",
    "style_axes", "style_legend",
    "line", "scatter", "bar", "box",
    "figure_width",
]
