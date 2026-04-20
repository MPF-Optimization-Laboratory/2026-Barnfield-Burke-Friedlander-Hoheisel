"""Shared matplotlib style for publication figures.

Usage::

    from selfscaling.figstyle import setup, loggrid, TEXTWIDTH
    setup()
"""

import matplotlib.pyplot as plt

TEXTWIDTH = 6.0  # inches; article class with 3cm margins on US letter


def setup():
    """Configure matplotlib for Computer Modern fonts and publication defaults."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,
    })


def loggrid(ax):
    """Add major and faint minor gridlines for log-scale axes."""
    ax.grid(True, which="major", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="minor", linewidth=0.3, alpha=0.15)
