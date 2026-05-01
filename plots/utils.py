# SPDX-License-Identifier: Apache-2.0
"""Backwards-compatibility shim.

The unified plotting style now lives in `plots/style.py`. This module
re-exports the functions and palettes scripts already imported via
`from utils import …`, so older callers keep working unchanged. New
code should import from `plots.style` directly.
"""
from style import (  # noqa: F401
    PALETTES,
    DEFAULT_PALETTE,
    MARKERS,
    LINE_STYLES,
    HATCHES,
    apply_style,
    palette,
    markers,
    line_styles,
    hatches,
    paper_figure,
    save_fig,
    style_axes,
    style_legend,
    line,
    scatter,
    bar,
    box,
)


# Legacy spellings kept for the old call sites.
def get_palette(n: int, name: str = "okabe_ito"):
    return palette(n, name)


def apply_color_cycle(n_series: int, name: str = "okabe_ito"):
    """Set the global color cycle. New code should call apply_style()
    which already does this."""
    import matplotlib.pyplot as plt
    from matplotlib import cycler
    colors = palette(n_series, name)
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
    return colors


def set_paper_style(*, base_font: float = 9.0,
                    palette_name: str = "okabe_ito",
                    **kwargs):
    """Legacy entry point. Forwards to apply_style()."""
    apply_style(base_font=base_font, palette_name=palette_name)
