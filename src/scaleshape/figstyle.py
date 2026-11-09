"""Shared matplotlib style for the Section 7 figures.

Every figure is drawn at the size it is printed: the paper (siamart, 10 pt)
includes each PDF at ``width=\\linewidth`` = 5.125 in, so a figure of width
``TEXTWIDTH`` is placed without scaling and its 8 pt text prints at 8 pt.

Conventions (shared with the figures of Friedlander, *Exact Sinkhorn*):
Computer Modern via LaTeX at 8 pt; L-shaped axes, inward ticks, no minor
ticks, no grid; Okabe-Ito colours, each paired with its own dash pattern so
that series survive grayscale and colour-vision deficiency; frameless
legends; "(a) Description" panel captions centred below each panel.

Usage::

    from scaleshape.figstyle import setup, TEXTWIDTH, series, panel_caption
    setup()
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator

TEXTWIDTH = 5.125  # inches; siamart \textwidth (one-column)

OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}

# Ordered sweeps (Z increasing) take these styles in order: colour and dash
# are redundant, so overlapping or grayscale-printed curves stay distinct.
SERIES = [
    dict(color=OKABE_ITO["orange"], linestyle="-"),
    dict(color=OKABE_ITO["sky"], linestyle=(0, (5, 2))),
    dict(color=OKABE_ITO["green"], linestyle=(0, (1.5, 1))),
    dict(color=OKABE_ITO["blue"], linestyle=(0, (5, 1, 1, 1))),
]
# Hollow markers for figures whose points are independent solves.
SERIES_MARKERS = ["o", "s", "^", "D"]

FONTSIZE = 8          # all figure text at its printed size
ANNOT_FONTSIZE = 8    # direct labels inside axes (one text size throughout)
REF_GRAY = "0.45"     # reference lines and their labels
# Unlabelled reference levels (stopping tolerance, targets): thin, light,
# solid, behind the data, so they never compete with dashed series.
LEVEL_STYLE = dict(color="0.75", linewidth=0.5, linestyle="-", zorder=0)


def setup():
    """Configure matplotlib for Computer Modern fonts and publication defaults."""
    plt.rcParams.update({
        # Fonts: the document's Computer Modern, via LaTeX.
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        # Layout: fixed figsize, no bbox cropping, so PDFs keep their width.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "savefig.dpi": 300,
        # Axes: left and bottom spines only, no grid.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "black",
        "axes.labelpad": 2,
        "axes.titlepad": 3,
        "axes.grid": False,
        "axes.prop_cycle": plt.cycler(color=[s["color"] for s in SERIES]),
        # Ticks: inward, major only.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        # Lines and markers.
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.6,
        "lines.dash_capstyle": "butt",
        "patch.linewidth": 0.6,
        # Legends: frameless, long enough handles to show the dash pattern.
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.6,
        "legend.labelspacing": 0.25,
        "legend.borderaxespad": 0.3,
        "legend.borderpad": 0.2,
    })


def series(i):
    """Line style (colour + dash) of the i-th member of an ordered sweep."""
    return dict(SERIES[i])


def log_ticks(ax, axis="y", every=1):
    """Major ticks at every ``every``-th decade, no minor ticks."""
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10.0 ** every, numticks=30))
    a.set_minor_locator(NullLocator())


def align_columns(fig, columns):
    """Freeze the layout and give the axes of each column one x-extent.

    Constrained layout sizes each subfigure on its own, so panels stacked in
    a column can end up with different widths. After one layout pass, each
    column takes the intersection of its axes' horizontal extents; the
    layout engine is then switched off so ``savefig`` keeps the positions.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for col in columns:
        boxes = [ax.get_window_extent() for ax in col]
        x0 = max(b.x0 for b in boxes)
        x1 = min(b.x1 for b in boxes)
        for ax, b in zip(col, boxes):
            to_parent = ax.get_figure(root=False).transSubfigure.inverted()
            (fx0, fy0), (fx1, fy1) = to_parent.transform([(x0, b.y0), (x1, b.y1)])
            ax.set_position([fx0, fy0, fx1 - fx0, fy1 - fy0])


def panel_caption(ax, text, pad=3):
    """Centre ``text`` (e.g. "(a) Merit") just below the axes' x-label."""
    ax.annotate(text, xy=(0.5, 0.0), xycoords=ax.xaxis.label,
                xytext=(0, -pad), textcoords="offset points",
                ha="center", va="top", fontsize=FONTSIZE)


# Termination markers drawn at the last iterate. Shape encodes the outcome
# (filled circle, filled down-triangle, cross, hollow square, as in the paper
# caption); the fill carries the series colour.
_END_SHAPE = {
    "converged": dict(marker="o", filled=True),
    "armijo_fatal": dict(marker="v", filled=True),
    "overflow_fatal": dict(marker="X", filled=True),
    "max_iters": dict(marker="s", filled=False),
}
END_MARKER_SIZE = 4.5  # points (diameter)


def endpoint_marker(ax, x, y, status, color):
    """Mark the final iterate (x, y) with the shape for ``status``."""
    shape = _END_SHAPE.get(status, _END_SHAPE["max_iters"])
    ax.plot([x], [y], linestyle="none", marker=shape["marker"],
            markersize=END_MARKER_SIZE,
            markerfacecolor=color if shape["filled"] else "white",
            markeredgecolor="black", markeredgewidth=0.6,
            zorder=10, clip_on=False)
