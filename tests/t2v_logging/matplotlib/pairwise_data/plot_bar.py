"""
Test cases for ``Axes.bar`` logging.

Strategy: computed
------------------
``patches_to_logdata`` logs the **top-centre** of each bar:

    logged_x[i] = patch.get_x() + patch.get_width() / 2
                = (x[i] - width/2) + width/2
                = x[i]                        ← original input x

    logged_y[i] = patch.get_y() + patch.get_height()
                = bottom + height
                = 0 + y[i]                    ← original input y (bottom=0)

So for standard bar charts with ``bottom=0`` (the default), the logged
coordinates equal the input values and expected values are computable.

Bars are grouped by face colour.  All bars in a single ``ax.bar()`` call share
the same default colour, so they produce **one series**.

Edge cases handled
------------------
- ``case_custom_bottom`` — ``bottom != 0``: ``logged_y = bottom + height``,
  NOT the raw height.  Expected is still computable if ``bottom`` is known.
- ``case_multiple_colors`` — each distinct colour produces a separate series.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Direct reproduction of the matplotlib bar-chart gallery example.

    Source (plt.style.use removed — does not affect data):
        x = 0.5 + np.arange(8)
        y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]
        ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
    """
    x = 0.5 + np.arange(8)
    y = np.array([4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0])

    ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    # All bars share the same default colour → one logged series.
    # logged x = top-centre x = original input x (width cancels out).
    # logged y = top-centre y = original input y (bottom=0 default).
    return [{"plt_func": "bar", "x": x, "y": y}]


def case_custom_bottom(ax):
    """
    Bar chart with a non-zero bottom — logged y is bottom + height, not height.
    """
    x      = np.array([1.0, 2.0, 3.0])
    height = np.array([2.0, 3.0, 1.5])
    bottom = np.array([1.0, 0.5, 2.0])

    ax.bar(x, height, bottom=bottom)

    # logged_y = bottom + height (top edge of each bar)
    return [{"plt_func": "bar", "x": x, "y": bottom + height}]


def case_uniform_width(ax):
    """Simple bars to verify x-centre and y-top reconstruction."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    ax.bar(x, y, width=0.6)

    return [{"plt_func": "bar", "x": x, "y": y}]
