"""
Test cases for ``Axes.errorbar`` logging.

Strategy: computed
------------------
``activate_axes_errorbar`` delegates to ``errorbarcontainer_to_logdata``:

    1. Data line (Line2D):        1 series  — x = input x,  y = input y
    2. Error-bar segments (LineCollection):  one series per data point
                                  x = [xᵢ, xᵢ],  y = [yᵢ-errᵢ, yᵢ+errᵢ]

Cap lines are intentionally skipped by the logger.
All values are directly computable from the inputs.

Series order
------------
The data line is always first, followed by the error-bar segments in input
order (one per data point, left to right).

Total series = 1  +  n_data_points
"""

import numpy as np


def _errorbar_segments(x, y, yerr=None, xerr=None):
    """
    Compute expected error-bar segment series.

    For y-errors: segment i = x=[xᵢ,xᵢ],  y=[yᵢ-dyᵢ, yᵢ+dyᵢ]
    For x-errors: segment i = x=[xᵢ-dxᵢ, xᵢ+dxᵢ], y=[yᵢ,yᵢ]
    """
    segments = []
    if yerr is not None:
        yerr = np.broadcast_to(yerr, len(y))
        for xi, yi, dyi in zip(x, y, yerr):
            segments.append({
                "plt_func": "errorbar",
                "x": [float(xi), float(xi)],
                "y": [float(yi - dyi), float(yi + dyi)],
            })
    if xerr is not None:
        xerr = np.broadcast_to(xerr, len(x))
        for xi, yi, dxi in zip(x, y, xerr):
            segments.append({
                "plt_func": "errorbar",
                "x": [float(xi - dxi), float(xi + dxi)],
                "y": [float(yi), float(yi)],
            })
    return segments


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib errorbar gallery example.

    Source (np.random.seed unused — data is hardcoded):
        x = [2, 4, 6]
        y = [3.6, 5, 4.2]
        yerr = [0.9, 1.2, 0.5]
        ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)

    Produces 4 series: data line + 3 y-error segments.
    """
    x    = [2.0, 4.0, 6.0]
    y    = [3.6, 5.0, 4.2]
    yerr = [0.9, 1.2, 0.5]

    ax.errorbar(x, y, yerr, fmt='o', linewidth=2, capsize=6)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [
        {"plt_func": "errorbar", "x": x, "y": y},   # data line
        *_errorbar_segments(x, y, yerr=yerr),         # y-error segments
    ]


def case_symmetric_yerr(ax):
    """Errorbar with a scalar (symmetric) y-error broadcast to all points."""
    x    = [1.0, 2.0, 3.0, 4.0]
    y    = [2.0, 3.5, 2.8, 4.1]
    yerr = 0.5   # scalar → same error for every point

    ax.errorbar(x, y, yerr=yerr, fmt='s')

    return [
        {"plt_func": "errorbar", "x": x, "y": y},
        *_errorbar_segments(x, y, yerr=np.full(len(x), yerr)),
    ]


def case_xerr_and_yerr(ax):
    """Errorbar with both x and y errors — produces 2n + 1 series total."""
    x    = [1.0, 3.0, 5.0]
    y    = [2.0, 4.0, 3.0]
    xerr = [0.3, 0.4, 0.2]
    yerr = [0.5, 0.6, 0.4]

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='D')

    # matplotlib creates x-error bars before y-error bars
    return [
        {"plt_func": "errorbar", "x": x, "y": y},
        *_errorbar_segments(x, y, xerr=xerr),   # x-segments first
        *_errorbar_segments(x, y, yerr=yerr),   # then y-segments
    ]
