"""
Test cases for ``Series.plot.bar`` / ``DataFrame.plot.bar`` logging.

Strategy: computed
------------------
Delegates to ``ax.bar()``.  Logged values (from ``patches_to_logdata``)
are the **top-centre** of each bar::

    logged_x = bar_position_center
    logged_y = bar_height  (= value, when bottom=0)

Series bar plot
~~~~~~~~~~~~~~~
All bars share one colour → **one logged series**.
Bar centres = integer tick positions [0, 1, ..., n-1] exactly::

    x = [0, 1, ..., n-1]
    y = series.values

DataFrame bar plot (grouped)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pandas uses ``total_width / n_cols`` per bar, offset by column index.
For ``total_width = 0.5`` and ``n_cols`` columns, the centre for
column ``k`` (0-indexed) at tick ``i`` is::

    x[i, k] = i + (k - n_cols/2 + 0.5) * (total_width / n_cols)

One logged series per colour (one colour per column).
"""

import numpy as np
import pandas as pd


def _bar_centers(n_rows: int, n_cols: int, col_idx: int,
                 total_width: float = 0.5) -> np.ndarray:
    """Compute bar centre x positions for a grouped DataFrame bar plot."""
    bar_width = total_width / n_cols
    ticks     = np.arange(n_rows, dtype=float)
    offset    = (col_idx - n_cols / 2 + 0.5) * bar_width
    return ticks + offset


def case_series_basic(ax):
    """Series bar — one series, x = integer positions, y = values."""
    s = pd.Series([1.0, 3.0, 2.0], index=["A", "B", "C"])
    s.plot.bar(ax=ax)
    return [{"plt_func": "bar",
             "x": np.arange(len(s), dtype=float),
             "y": s.values}]


def case_dataframe_grouped(ax):
    """DataFrame bar — one series per column with offset centres."""
    df = pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0]},
        index=["x", "y"],
    )
    df.plot.bar(ax=ax)
    return [
        {"plt_func": "bar",
         "x": _bar_centers(2, 2, 0), "y": df["A"].values},
        {"plt_func": "bar",
         "x": _bar_centers(2, 2, 1), "y": df["B"].values},
    ]


def case_series_from_docs(ax):
    """
    Bar chart from the pandas visualization docs:
        df2 = pd.DataFrame(np.random.rand(10, 4), columns=["a","b","c","d"])
        df2.iloc[0].plot.bar()   ← single row becomes a Series
    """
    np.random.seed(123456)
    df2  = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
    row  = df2.iloc[0]   # Series with index ["a","b","c","d"]
    row.plot.bar(ax=ax)
    return [{"plt_func": "bar",
             "x": np.arange(len(row), dtype=float),
             "y": row.values}]
