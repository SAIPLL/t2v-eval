"""
Test cases for ``Axes.stem`` logging.

Strategy: computed (partial for baseline)
------------------------------------------
``activate_axes_stem`` logs three artist groups in a single JSON file, in this
fixed order:

    1. markerline  (Line2D)          → 1 series,  x=input x, y=input y
    2. stemlines   (LineCollection)  → N series,  x=[xᵢ,xᵢ], y=[0, yᵢ]
    3. baseline    (Line2D)          → 1 series,  y=[0, 0], x=extent (uncertain)

Total series = 1 + N + 1  (where N = number of data points).

Markerline and stemlines are fully computable from inputs.
The baseline x-extent depends on the matplotlib version (it may be
``[x[0], x[-1]]`` or the full locs array), so only ``plt_func`` is checked
for the baseline series — coordinate values are intentionally omitted.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Stem chart from the matplotlib gallery example.

    Source (plt.style.use removed — does not affect data):
        x = 0.5 + np.arange(8)
        y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]
        ax.stem(x, y)
    """
    x = 0.5 + np.arange(8)
    y = np.array([4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0])

    ax.stem(x, y)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [
        # 1. markerline — data points at stem tops
        {"plt_func": "stem", "x": x, "y": y},
        # 2-9. stemlines — one vertical segment per point: (xᵢ, 0) → (xᵢ, yᵢ)
        *[
            {"plt_func": "stem", "x": [xi, xi], "y": [0.0, yi]}
            for xi, yi in zip(x, y)
        ],
        # 10. baseline — y=0 horizontal line; x-extent is version-dependent
        {"plt_func": "stem"},   # only checks plt_func + length consistency
    ]


def case_simple(ax):
    """Minimal stem plot to verify the 1 + N + 1 series structure."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 5.0, 3.0])

    ax.stem(x, y)

    return [
        # markerline
        {"plt_func": "stem", "x": x, "y": y},
        # stemlines
        *[
            {"plt_func": "stem", "x": [xi, xi], "y": [0.0, yi]}
            for xi, yi in zip(x, y)
        ],
        # baseline (value check skipped)
        {"plt_func": "stem"},
    ]


def case_custom_bottom(ax):
    """Stem with a non-zero bottom — stemlines start at bottom, not 0."""
    x      = np.array([1.0, 2.0, 3.0])
    y      = np.array([4.0, 3.0, 5.0])
    bottom = 1.0

    ax.stem(x, y, bottom=bottom)

    return [
        # markerline
        {"plt_func": "stem", "x": x, "y": y},
        # stemlines: (xᵢ, bottom) → (xᵢ, yᵢ)
        *[
            {"plt_func": "stem", "x": [xi, xi], "y": [bottom, yi]}
            for xi, yi in zip(x, y)
        ],
        # baseline at y=bottom (value check skipped)
        {"plt_func": "stem"},
    ]
