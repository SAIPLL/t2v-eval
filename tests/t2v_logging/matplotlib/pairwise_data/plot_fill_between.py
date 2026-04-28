"""
Test cases for ``Axes.fill_between`` logging.

Strategy: computed + sorted-pair comparison
--------------------------------------------
``polycollection_to_logdata`` logs all 2n+3 polygon vertices.  The multiset of
``(x, y)`` pairs equals::

    list(x)  + list(x)  + [x[-1],   x[0],   x[0]]   ← x repeated
    list(y1) + list(y2) + [y2[-1], y2[0], y2[0]]     ← both curves + extras

Because the polygon vertex order differs from input order, comparisons use
``"_sort": True`` (via :func:`~tests.t2v_logging.helpers.fill_between_expected`).
"""

import numpy as np

from tests.t2v_logging.helpers import fill_between_expected


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib fill_between gallery example.

    Source (plt.style.use removed):
        np.random.seed(1)
        x  = np.linspace(0, 8, 16)
        y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
        y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))
        ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
        ax.plot(x, (y1 + y2) / 2, linewidth=2)
    """
    np.random.seed(1)
    x  = np.linspace(0, 8, 16)
    y1 = 3 + 4 * x / 8 + np.random.uniform(0.0, 0.5, len(x))
    y2 = 1 + 2 * x / 8 + np.random.uniform(0.0, 0.5, len(x))

    ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
    ax.plot(x, (y1 + y2) / 2, linewidth=2)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [
        fill_between_expected(x, y1, y2),
        {"plt_func": "plot", "x": x, "y": (y1 + y2) / 2},
    ]


def case_fill_between_only(ax):
    """Standalone fill_between with a simple deterministic input."""
    x  = np.linspace(0, 5, 10)
    y1 = np.sin(x) + 2
    y2 = np.cos(x) + 2

    ax.fill_between(x, y1, y2, alpha=0.4)

    return [fill_between_expected(x, y1, y2)]


def case_fill_between_and_plot_separate(ax):
    """Two fill_between calls + one plot — verifies series ordering."""
    x  = np.linspace(0, 4, 8)
    y1 = x ** 2 / 4
    y2 = x / 2
    y3 = x ** 2 / 8

    ax.fill_between(x, y1, y2, alpha=0.3)
    ax.fill_between(x, y2, y3, alpha=0.3)
    ax.plot(x, (y1 + y3) / 2)

    return [
        fill_between_expected(x, y1, y2),
        fill_between_expected(x, y2, y3),
        {"plt_func": "plot", "x": x, "y": (y1 + y3) / 2},
    ]
