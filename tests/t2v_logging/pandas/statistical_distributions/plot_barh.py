"""
Test cases for ``Series.plot.barh`` logging.

Strategy: computed
------------------
Delegates to ``ax.barh()``.  For horizontal bars, ``patches_to_logdata``
computes the **right-centre** of each bar::

    logged_x = bar_width   (= value, when left=0)
    logged_y = bar_position_center  (= integer tick [0, 1, ..., n-1])

For a Series with n values, all bars share one colour → **one series**.
"""

import numpy as np
import pandas as pd


def case_series_basic(ax):
    """Series horizontal bar — x = values, y = integer positions."""
    s = pd.Series([1.0, 3.0, 2.0], index=["A", "B", "C"])
    s.plot.barh(ax=ax)
    return [{"plt_func": "barh",
             "x": s.values,
             "y": np.arange(len(s), dtype=float)}]


def case_series_from_docs(ax):
    """
    Horizontal bar from the pandas docs:
        df2.plot.barh(stacked=True)
    Use a single row (Series) for simplicity.
    """
    np.random.seed(123456)
    df2 = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
    row = df2.iloc[0]
    row.plot.barh(ax=ax)
    return [{"plt_func": "barh",
             "x": row.values,
             "y": np.arange(len(row), dtype=float)}]
