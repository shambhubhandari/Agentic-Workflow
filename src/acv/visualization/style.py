"""System: style module.

Provides strict, deterministic logic and strict typing for style operations.
"""
# =============================================================================
#                    ********* MANUSCRIPT FIGURES *********                    
#                         Strict definitions for style.                        
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Column widths of the manuscript, in inches, read off its own LaTeX run:
#   \columnwidth = 222.62 pt, \textwidth = 455.24 pt (10 pt, a4paper, twocolumn,
#   2.5 cm margins).  Figures are authored at exactly these sizes and included with
#   width=\columnwidth / width=\textwidth, so LaTeX scales them by 1.0 and the 7 pt
#   type in the figure is 7 pt on the page.
COL_SINGLE = 222.62 / 72
COL_DOUBLE = 455.24 / 72

# Every plotting box in the manuscript is this size, in inches.  Panel geometry is
# fixed in absolute units and the canvas is fixed to a column width, so no figure is
# rescaled on the page: one panel is one panel, at one type size, in every figure.
PANEL = 2.05


def setup_plotting_style():
    """Configure matplotlib rcParams for two-column journal figure style (sans-serif, 300 dpi)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Use metric-compatible font.
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans",
                            "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",
        "figure.titleweight": "bold",
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        # Disable tight bounding to preserve precise panel geometry.
        "savefig.bbox": "standard",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "text.usetex": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def panel_grid(ncols: int = 1, fig_w: float = COL_SINGLE, ax_w: float = PANEL,
               ax_h: float = PANEL, left: float = 0.60, gap: float = 0.62,
               bottom: float = 0.46, top: float = 0.12):
    """Figure with `ncols` panels of exactly ax_w x ax_h inches on a fixed canvas.

    All margins are inches.  The right margin is whatever is left over, so the canvas
    width stays exactly a journal column width while the drawing boxes stay identical
    from figure to figure.
    """
    fig_h = bottom + ax_h + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes([(left + i * (ax_w + gap)) / fig_w, bottom / fig_h,
                          ax_w / fig_w, ax_h / fig_h]) for i in range(ncols)]
    return fig, axes


def format_axes(ax, minor_x: int = 4, minor_y: int = 4, grid_axis: str = "both"):
    """Boxed axes with minor ticks on all four sides and a faint dotted grid."""
    ax.tick_params(which="both", top=True, right=True, direction="in")
    if minor_x:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(minor_x))
    if minor_y:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(minor_y))
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, ls=":", lw=0.4, alpha=0.25)
    ax.set_axisbelow(True)


def save_plot(fig, path):
    """Save figure to path, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".pdf":
        fig.savefig(path, backend="pdf")
    else:
        fig.savefig(path, dpi=300)
    return path
