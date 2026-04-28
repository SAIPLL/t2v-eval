"""
Test cases for ``Axes.fill_betweenx`` logging.

Strategy: computed + sorted-pair comparison
--------------------------------------------
``fill_betweenx(y, x1, x2)`` fills between two x-boundary curves as a
function of y.  ``polycollection_to_logdata`` records the same 2n+3
polygon vertices as ``fill_between``, but with x and y axes swapped::

    Vertex order (logged):
        MOVETO  (x2[0], y[0])
        LINETO  (x1[0..n-1], y[0..n-1])   ← forward along left boundary
        LINETO  (x2[n-1], y[n-1]) × 2     ← duplicate at top
        LINETO  (x2[n-2..0], y[n-2..0])   ← backward along right boundary
        LINETO  + CLOSEPOLY (x2[0], y[0]) × 2

    Expected multiset formula (same as fill_between with axes swapped):
        exp_x = list(x1) + list(x2) + [x2[-1], x2[0], x2[0]]
        exp_y = list(y)  + list(y)  + [y[-1],  y[0],  y[0] ]

``"_sort": True`` handles the ordering via
:func:`~tests.t2v_logging.helpers.fill_betweenx_expected`.
"""

import numpy as np

from tests.t2v_logging.helpers import fill_betweenx_expected


def case_basic(ax):
    """Simple fill_betweenx between two x-boundary curves."""
    y  = np.array([0.0, 1.0, 2.0, 3.0])
    x1 = np.array([1.0, 2.0, 1.5, 2.5])
    x2 = np.array([3.0, 4.0, 3.5, 4.5])

    ax.fill_betweenx(y, x1, x2, alpha=0.5)

    return [fill_betweenx_expected(y, x1, x2)]


def case_scalar_x1(ax):
    """Scalar left boundary — x1 broadcast to match y length."""
    y  = np.array([0.0, 1.0, 2.0])
    x2 = np.array([1.0, 2.0, 1.5])

    ax.fill_betweenx(y, 0.0, x2)   # x1 = 0 (scalar)

    x1 = np.zeros_like(y)
    return [fill_betweenx_expected(y, x1, x2)]


def case_with_plot(ax):
    """
    fill_betweenx combined with a midline plot — verifies series ordering.
    Two series: fill_betweenx first, then plot.
    """
    y    = np.linspace(0, 4, 8)
    x1   = y ** 2 / 4
    x2   = y + 1
    xmid = (x1 + x2) / 2

    ax.fill_betweenx(y, x1, x2, alpha=0.3)
    ax.plot(xmid, y)

    return [
        fill_betweenx_expected(y, x1, x2),
        {"plt_func": "plot", "x": xmid, "y": y},
    ]
