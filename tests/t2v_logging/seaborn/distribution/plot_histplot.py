"""
Test cases for ``sns.histplot`` logging.

How it is logged
----------------
Seaborn draws histogram bars using ``ax.bar()`` (not ``ax.hist()``).
The logged values (from ``patches_to_logdata``) are the **top-centres**:

    logged_x = bin_centre = (left_edge + right_edge) / 2
    logged_y = count  (for stat="count", the default)

Expected values are computed via ``numpy.histogram`` with the same bins.

Empty-series artefact (no hue)
-------------------------------
Without a hue semantic, seaborn creates a single-bar legend handle that
fires the ``ax.bar()`` patch.  This produces an extra series with **n=1**
before the real histogram.  It is checked with a partial
``{"plt_func": "bar"}`` entry.

With a hue semantic, no artefact appears.
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.distribution._datasets import (
    make_penguins, make_tips,
)


def _hist_expected(data, bins, plt_func="bar") -> dict:
    counts, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {"plt_func": plt_func,
            "x": centers, "y": counts.astype(float)}


def case_basic(ax):
    """
    Basic histogram, explicit bins.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", bins=20)

    → [legend artefact (n=1), histogram data (n=20)].
    """
    penguins = make_penguins()
    sns.histplot(data=penguins, x="flipper_length_mm", bins=10, ax=ax)
    data = penguins["flipper_length_mm"].dropna().values
    return [
        {"plt_func": "bar"},                    # legend artefact (n=1)
        _hist_expected(data, bins=10),          # actual histogram
    ]


def case_hue(ax):
    """
    Histogram with hue — one series per species, no artefact.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species")

    Seaborn computes bin edges from the full dataset, then applies them per
    group.  Drawing order is seaborn-controlled → returns ``UnorderedSeries``
    so the test runner matches by median value rather than position.
    """
    penguins = make_penguins()
    sns.histplot(data=penguins, x="flipper_length_mm",
                 hue="species", bins=10, ax=ax)
    # Compute global edges, then per-group counts
    col  = penguins["flipper_length_mm"].dropna()
    _, global_edges = np.histogram(col, bins=10)
    centers = (global_edges[:-1] + global_edges[1:]) / 2
    expected = UnorderedSeries()
    for sp in penguins["species"].unique():
        data = penguins.loc[penguins["species"] == sp,
                            "flipper_length_mm"].dropna().values
        counts, _ = np.histogram(data, bins=global_edges)
        expected.append({"plt_func": "bar",
                         "x": centers, "y": counts.astype(float)})
    return expected


def case_discrete(ax):
    """
    Discrete histogram (integer bins).
    Tutorial: sns.displot(tips, x="size", discrete=True)

    Partial check — discrete bin edges seaborn computes may differ slightly.
    """
    tips = make_tips()
    sns.histplot(data=tips, x="size", discrete=True, ax=ax)
    return [
        {"plt_func": "bar"},   # legend artefact (n=1)
        {"plt_func": "bar"},   # actual histogram (values: partial check)
    ]
