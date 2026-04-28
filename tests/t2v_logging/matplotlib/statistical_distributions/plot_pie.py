"""
Test cases for ``Axes.pie`` logging.

Strategy: computed
------------------
``activate_axes_pie`` logs the raw slice values and labels directly from
the call arguments — no internal transformation is applied::

    logged x      = args[0]          ← raw input values (not normalised)
    logged labels = kwargs["labels"]  ← None if omitted

The normalisation to proportions (÷ sum) happens later in
``parse_t2v_log_dir`` at evaluation time, not at logging time.

Logged structure (one series)
------------------------------
    plt_func = "pie"
    x        = input x values
    labels   = label list or None

Note: ``labels`` is metadata and is not checked by ``assert_series_matches``
(it is not a coordinate key).  Only ``x`` is verified for value accuracy.
"""

import matplotlib.pyplot as plt
import numpy as np


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib pie gallery example.

    Source (plt.style.use removed):
        x = [1, 2, 3, 4]
        ax.pie(x, colors=colors, radius=3, center=(4, 4), ...)

    Raw values [1, 2, 3, 4] are logged — not the proportions [10%, 20%, 30%, 40%].
    """
    x      = [1, 2, 3, 4]
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(x)))

    ax.pie(x, colors=colors, radius=3, center=(4, 4),
           wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [{"plt_func": "pie", "x": x}]


def case_with_labels(ax):
    """Pie with explicit labels — labels are stored in the log but not checked."""
    x      = [30, 20, 25, 25]
    labels = ["A", "B", "C", "D"]

    ax.pie(x, labels=labels)

    return [{"plt_func": "pie", "x": x}]


def case_single_slice(ax):
    """Edge case — single slice (100% pie)."""
    x = [1]

    ax.pie(x)

    return [{"plt_func": "pie", "x": x}]


def case_float_values(ax):
    """Pie with float slice values — logged as-is without rounding."""
    x = [0.1, 0.25, 0.4, 0.25]

    ax.pie(x)

    return [{"plt_func": "pie", "x": x}]
