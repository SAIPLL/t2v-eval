"""
Test cases for ``Series.plot`` / ``DataFrame.plot`` (line) logging.

Strategy: computed
------------------
Pandas line plots delegate to ``ax.plot()``.  The logged values are::

    plt_func = "plot"
    x        = index values as float   (RangeIndex → [0., 1., 2., ...])
    y        = column / series values

One series per column is produced.
"""

import numpy as np
import pandas as pd


def case_series_basic(ax):
    """Single Series with a default integer index."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    s.plot(ax=ax)
    return [{"plt_func": "plot",
             "x": s.index.values.astype(float), "y": s.values}]


def case_dataframe_two_columns(ax):
    """DataFrame with two columns — one series logged per column."""
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]})
    df.plot(ax=ax)
    x = df.index.values.astype(float)
    return [
        {"plt_func": "plot", "x": x, "y": df["A"].values},
        {"plt_func": "plot", "x": x, "y": df["B"].values},
    ]


def case_series_float_index(ax):
    """Series with a custom float index — x = index, y = values."""
    idx = np.array([0.5, 1.0, 2.0, 4.0])
    s   = pd.Series([3.0, 1.0, 4.0, 1.5], index=idx)
    s.plot(ax=ax)
    return [{"plt_func": "plot", "x": idx, "y": s.values}]
