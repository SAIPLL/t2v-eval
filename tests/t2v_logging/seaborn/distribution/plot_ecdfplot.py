"""
Test cases for ``sns.ecdfplot`` logging.

How it is logged
----------------
ECDF is drawn with ``ax.plot()`` → plt_func = "plot".

The step function starts at **x = -inf, y = 0**, then each data point
adds one step.  The logged values are::

    x = [-inf, sorted_data[0], sorted_data[1], ..., sorted_data[n-1]]
    y = [0, 1/n, 2/n, ..., 1.0]

Strategy: computed
------------------
Both x and y are fully derivable from the input data (sorted + prepend -inf).

Empty-series artefact
---------------------
Without hue: empty (n=0) "plot" series precedes the ECDF series.
With hue: no artefact.
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.distribution._datasets import make_penguins


def _ecdf_expected(data) -> dict:
    """Compute expected ECDF series for a 1-D array."""
    arr = np.sort(data[~np.isnan(data)])
    n   = len(arr)
    x   = np.concatenate([[-np.inf], arr])
    y   = np.arange(0, n + 1) / n
    return {"plt_func": "plot", "x": x, "y": y}


def case_basic(ax):
    """
    Basic ECDF.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", kind="ecdf")

    → [empty "plot", ECDF "plot"].
    """
    penguins = make_penguins()
    sns.ecdfplot(data=penguins, x="flipper_length_mm", ax=ax)
    data = penguins["flipper_length_mm"].values
    return [
        {"plt_func": "plot"},           # empty artefact
        _ecdf_expected(data),           # ECDF line (values checked)
    ]


def case_hue(ax):
    """
    ECDF per species — no artefact.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf")

    Drawing order is seaborn-controlled → ``UnorderedSeries`` with full values.
    Each group's ECDF is independently computable from its sorted data.
    """
    penguins = make_penguins()
    sns.ecdfplot(data=penguins, x="flipper_length_mm", hue="species", ax=ax)
    expected = UnorderedSeries()
    for sp in penguins["species"].unique():
        data = penguins.loc[penguins["species"] == sp,
                            "flipper_length_mm"].values
        expected.append(_ecdf_expected(data))
    return expected
