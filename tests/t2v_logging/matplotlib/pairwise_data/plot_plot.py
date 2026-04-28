"""
Test cases for ``Axes.plot`` logging.

Strategy: computed
------------------
For simple numeric data on a linear axes, ``line2d.get_xdata(orig=False)``
returns the same values as the input, so expected values are computed directly.

One edge case to be aware of
-----------------------------
A ``plot`` call with no linestyle (e.g. marker-only format string ``'x'``) is
still logged as ``plt_func="plot"`` with ``linestyle="None"``.  The
reclassification to ``"scatter"`` happens later in ``parse_t2v_log_dir``
(evaluation time), NOT at logging time.  So the expected plt_func here is
always ``"plot"``.
"""

import numpy as np


def case_single_line(ax):
    """Single solid line — the most basic plot call."""
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    ax.plot(x, y)
    return [{"plt_func": "plot", "x": x, "y": y}]


def case_three_styles(ax):
    """
    Three ax.plot calls with different styles — from the matplotlib gallery.

    Replicates:
        ax.plot(x2, y2 + 2.5, 'x', markeredgewidth=2)   # marker only
        ax.plot(x,  y,          linewidth=2.0)            # solid line
        ax.plot(x2, y2 - 2.5,  'o-', linewidth=2)        # line + marker

    All three are logged as plt_func="plot" (reclassification to scatter
    happens only at evaluation time).  xlim/ylim via ax.set() do not affect
    the stored data values.
    """
    x  = np.linspace(0, 10, 100)
    y  = 4 + 1 * np.sin(2 * x)
    x2 = np.linspace(0, 10, 25)
    y2 = 4 + 1 * np.sin(2 * x2)

    ax.plot(x2, y2 + 2.5, 'x', markeredgewidth=2)
    ax.plot(x,  y,          linewidth=2.0)
    ax.plot(x2, y2 - 2.5,  'o-', linewidth=2)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [
        {"plt_func": "plot", "x": x2, "y": y2 + 2.5},
        {"plt_func": "plot", "x": x,  "y": y},
        {"plt_func": "plot", "x": x2, "y": y2 - 2.5},
    ]


def case_multiple_series_same_axes(ax):
    """Two lines on the same axes — verifies both are captured separately."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    ax.plot(x, x ** 2)
    ax.plot(x, x ** 3)
    return [
        {"plt_func": "plot", "x": x, "y": x ** 2},
        {"plt_func": "plot", "x": x, "y": x ** 3},
    ]
