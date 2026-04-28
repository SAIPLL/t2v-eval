"""
Test cases for ``Axes.barh`` logging.

Strategy: computed
------------------
``patches_to_logdata`` with ``orientation="horizontal"`` logs the
**right-centre** of each bar::

    logged_x = patch.get_x() + patch.get_width()   = left + width   (right edge)
    logged_y = patch.get_y() + patch.get_height()/2 = bar y-centre

For the default ``left=0``:
    logged_x = width  (the bar value itself)
    logged_y = y_position

For a non-zero ``left=L``:
    logged_x = L + width  (absolute right edge)
    logged_y = y_position

All bars sharing the same face colour form **one series**.
"""

import numpy as np


def case_basic(ax):
    """Simple horizontal bars — logged x = widths, y = positions."""
    y      = np.array([1.0, 2.0, 3.0])
    widths = np.array([2.0, 3.0, 1.0])

    ax.barh(y, widths)

    return [{"plt_func": "barh", "x": widths, "y": y}]


def case_non_zero_left(ax):
    """
    Bars with a non-zero left baseline — logged x = left + width,
    NOT just width.
    """
    y      = np.array([1.0, 2.0, 3.0])
    widths = np.array([2.0, 3.0, 1.0])
    left   = 1.0

    ax.barh(y, widths, left=left)

    return [{"plt_func": "barh", "x": left + widths, "y": y}]


def case_single_bar(ax):
    """Single horizontal bar — minimal case."""
    ax.barh([2.0], [5.0])
    return [{"plt_func": "barh",
             "x": np.array([5.0]), "y": np.array([2.0])}]
