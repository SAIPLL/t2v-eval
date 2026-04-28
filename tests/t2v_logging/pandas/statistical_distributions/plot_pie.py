"""
Test cases for ``Series.plot.pie`` logging.

Strategy: computed
------------------
Delegates to ``ax.pie()``.  The raw slice values are logged as-is::

    plt_func = "pie"
    x        = series.values  (raw, not normalised to proportions)

Identical strategy to the matplotlib pie tests.
"""

import numpy as np
import pandas as pd


def case_series_from_docs(ax):
    """
    Pie from the pandas docs:
        series = pd.Series(3*np.random.rand(4), index=["a","b","c","d"])
        series.plot.pie()
    """
    np.random.seed(123456)
    s = pd.Series(
        3 * np.random.rand(4), index=["a", "b", "c", "d"], name="series"
    )
    s.plot.pie(ax=ax)
    return [{"plt_func": "pie", "x": s.values}]


def case_integer_values(ax):
    """Simple integer slice values — logged without normalisation."""
    s = pd.Series([10, 20, 30, 40], index=["W", "X", "Y", "Z"])
    s.plot.pie(ax=ax)
    return [{"plt_func": "pie", "x": s.values.astype(float)}]


def case_subunit_values(ax):
    """
    Values that sum to < 1 — pandas passes raw values to ax.pie unchanged.
    The logged x = raw series values (NOT rescaled).
    """
    s = pd.Series([0.1, 0.1, 0.1, 0.1],
                  index=["a", "b", "c", "d"], name="series2")
    s.plot.pie(ax=ax)
    return [{"plt_func": "pie", "x": s.values}]
