"""
Test cases for ``Axes.violinplot`` logging.

Strategy: computed
------------------
The logging pipeline for ``violinplot`` is:

    1. ``activate_axes_violinplot`` intercepts the call and injects
       ``quantiles=[[0, 0.25, 0.5, 0.75, 1]] × n_violins`` before
       delegating to the original ``violinplot``.

    2. The original ``violinplot`` calls ``violin`` internally, which IS
       monkey-patched by ``activate_axes_violin``.

    3. ``activate_axes_violin`` extracts midpoints from the ``cquantiles``
       LineCollection (the five quantile tick marks) via
       ``get_mid_points_of_linecollection`` and logs them as ``"bxp"`` data.

Logged structure (one series per violin)
-----------------------------------------
    plt_func = "bxp"
    x        = [position] × 5
    y        = sorted([min, Q1, median, Q3, max])
             = np.percentile(col, [0, 25, 50, 75, 100])

This matches the boxplot (bxp) series structure, but y covers the full data
range (0th–100th percentile) rather than 1.5 × IQR whiskers.

Note on ``showextrema=False``
------------------------------
Setting ``showextrema=False`` suppresses the min/max bar lines, but does NOT
suppress the injected quantile ticks.  The 5 quantile points (including 0th
and 100th) are always logged regardless of ``showextrema``.
"""

import numpy as np


def _violin_expected(col: np.ndarray, position: float) -> dict:
    """
    Compute the expected ``bxp`` series logged for one violin.

    Parameters
    ----------
    col : ndarray
        1-D data array for a single violin.
    position : float
        x-position of the violin.
    """
    q = np.percentile(col, [0, 25, 50, 75, 100])
    return {
        "plt_func": "bxp",
        "x": [float(position)] * 5,
        "y": [float(v) for v in q],   # already sorted: min → max
    }


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib violinplot gallery example.

    Source (plt.style.use removed):
        np.random.seed(10)
        D = np.random.normal((3, 5, 4), (0.75, 1.00, 0.75), (200, 3))
        ax.violinplot(D, [2, 4, 6], widths=2,
                      showmeans=False, showmedians=False, showextrema=False)

    Three violins → three logged series, one per position.
    """
    np.random.seed(10)
    D = np.random.normal((3, 5, 4), (0.75, 1.00, 0.75), (200, 3))

    vp = ax.violinplot(D, [2, 4, 6], widths=2,
                       showmeans=False, showmedians=False, showextrema=False)
    for body in vp['bodies']:
        body.set_alpha(0.9)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [
        _violin_expected(D[:, 0], position=2),
        _violin_expected(D[:, 1], position=4),
        _violin_expected(D[:, 2], position=6),
    ]


def case_single_violin(ax):
    """Single violin at the default position."""
    np.random.seed(0)
    data = np.random.normal(5, 1, 100)

    ax.violinplot([data])

    return [_violin_expected(data, position=1)]


def case_custom_positions(ax):
    """Two violins at non-default positions."""
    np.random.seed(3)
    col0 = np.random.normal(3, 0.5, 80)
    col1 = np.random.normal(7, 1.0, 80)

    ax.violinplot([col0, col1], positions=[2, 8])

    return [
        _violin_expected(col0, position=2),
        _violin_expected(col1, position=8),
    ]
