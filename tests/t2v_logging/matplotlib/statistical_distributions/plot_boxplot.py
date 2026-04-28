"""
Test cases for ``Axes.boxplot`` logging.

Strategy: computed
------------------
``boxplot`` calls ``bxp`` internally, which IS monkey-patched.
``activate_axes_bxp`` collects two types of artists per box:

    1 median point  → (position, median_value)
    4 whisker points → (position, Q1), (position, whislo),
                       (position, Q3),  (position, whishi)

These 5 points are grouped by x-position and sorted ascending by y, giving
one series per box with::

    x = [position] × 5
    y = sorted([whislo, Q1, median, Q3, whishi])

All values are computable via ``numpy.percentile`` with the same 1.5×IQR
whisker rule that matplotlib uses internally.

Whisker endpoints
-----------------
    whislo = min(data[data >= Q1 - 1.5·IQR])   ← lowest non-outlier
    whishi = max(data[data <= Q3 + 1.5·IQR])   ← highest non-outlier
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _bxp_expected(col: np.ndarray, position: float, whis: float = 1.5) -> dict:
    """
    Compute the expected ``bxp`` series for one data column.

    Parameters
    ----------
    col : ndarray
        1-D data array for a single box.
    position : float
        x-position of the box.
    whis : float
        Whisker length as a multiple of IQR (default 1.5).
    """
    q1, median, q3 = np.percentile(col, [25, 50, 75])
    iqr    = q3 - q1
    whislo = col[col >= q1 - whis * iqr].min()
    whishi = col[col <= q3 + whis * iqr].max()

    y_sorted = sorted([float(whislo), float(q1),
                       float(median), float(q3), float(whishi)])
    return {
        "plt_func": "bxp",
        "x": [float(position)] * 5,
        "y": y_sorted,
    }


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def case_from_gallery(ax):
    """
    Reproduction of the matplotlib boxplot gallery example.

    Source (plt.style.use removed):
        np.random.seed(10)
        D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))
        ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
                   showmeans=False, showfliers=False, ...)

    Three boxes → three logged series, one per position.
    """
    np.random.seed(10)
    D = np.random.normal((3, 5, 4), (1.25, 1.00, 1.25), (100, 3))

    ax.boxplot(D, positions=[2, 4, 6], widths=1.5, patch_artist=True,
               showmeans=False, showfliers=False,
               medianprops={"color": "white", "linewidth": 0.5},
               boxprops={"facecolor": "C0", "edgecolor": "white",
                         "linewidth": 0.5},
               whiskerprops={"color": "C0", "linewidth": 1.5},
               capprops={"color": "C0", "linewidth": 1.5})
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [
        _bxp_expected(D[:, 0], position=2),
        _bxp_expected(D[:, 1], position=4),
        _bxp_expected(D[:, 2], position=6),
    ]


def case_single_box(ax):
    """Single box at default position 1 — minimal case."""
    np.random.seed(0)
    data = np.random.normal(5, 1, 50)

    ax.boxplot(data, showfliers=False)

    return [_bxp_expected(data, position=1)]


def case_custom_whis(ax):
    """Box with a non-default whisker length (whis=2.0)."""
    np.random.seed(7)
    data = np.random.normal(4, 1.5, 80)

    ax.boxplot(data, whis=2.0, showfliers=False, positions=[3])

    return [_bxp_expected(data, position=3, whis=2.0)]
