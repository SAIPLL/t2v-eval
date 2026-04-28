"""
Test cases for ``Axes.hist`` logging.

Strategy: computed
------------------
``activate_axes_hist`` uses ``patches_to_logdata`` — the same helper as
``bar`` — so the logged values are the **top-centre** of each bar:

    logged_x[i] = bin_left[i] + bin_width[i] / 2  = bin_centre[i]
    logged_y[i] = 0 + count[i]                    = count[i]

Both are computable via ``numpy.histogram``, which returns the same bins and
counts that matplotlib uses internally.

All bars share the default colour → **one logged series**.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib histogram gallery example.

    Source (plt.style.use removed):
        np.random.seed(1)
        x = 4 + np.random.normal(0, 1.5, 200)
        ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")
    """
    np.random.seed(1)
    x = 4 + np.random.normal(0, 1.5, 200)

    ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 56), yticks=np.linspace(0, 56, 9))

    # matplotlib uses the same computation as numpy.histogram internally
    counts, edges = np.histogram(x, bins=8)
    centers = (edges[:-1] + edges[1:]) / 2

    return [{"plt_func": "hist", "x": centers, "y": counts.astype(float)}]


def case_explicit_range(ax):
    """Histogram with explicit range — bin edges are determined by range, not data."""
    np.random.seed(42)
    x = np.random.normal(5, 1, 100)

    ax.hist(x, bins=5, range=(2, 8))

    counts, edges = np.histogram(x, bins=5, range=(2, 8))
    centers = (edges[:-1] + edges[1:]) / 2

    return [{"plt_func": "hist", "x": centers, "y": counts.astype(float)}]


def case_uniform_data(ax):
    """Histogram of uniform data — predictable bin counts."""
    x = np.linspace(0, 10, 100)   # uniform, no randomness

    ax.hist(x, bins=5)

    counts, edges = np.histogram(x, bins=5)
    centers = (edges[:-1] + edges[1:]) / 2

    return [{"plt_func": "hist", "x": centers, "y": counts.astype(float)}]
