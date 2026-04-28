"""
Test cases for ``DataFrame.plot.box`` logging.

Strategy: computed
------------------
Delegates to ``ax.bxp()``.  Same formula as the matplotlib boxplot tests::

    plt_func = "bxp"
    x        = [position] × 5
    y        = sorted([whislo, Q1, median, Q3, whishi])

Key difference from matplotlib: pandas assigns **1-indexed** positions
(column 0 → position 1, column 1 → position 2, …) while the matplotlib
test used caller-supplied positions.
"""

import numpy as np
import pandas as pd


def _bxp_expected(col: np.ndarray, position: float, whis: float = 1.5) -> dict:
    """Expected bxp series for one DataFrame column."""
    q1, med, q3 = np.percentile(col, [25, 50, 75])
    iqr    = q3 - q1
    whislo = col[col >= q1 - whis * iqr].min()
    whishi = col[col <= q3 + whis * iqr].max()
    y = sorted([float(whislo), float(q1), float(med), float(q3), float(whishi)])
    return {"plt_func": "bxp", "x": [float(position)] * 5, "y": y}


def case_two_columns(ax):
    """
    Two-column DataFrame — two bxp series at positions 1 and 2.

    From the pandas docs:
        df = pd.DataFrame(np.random.rand(10, 5), columns=["A","B","C","D","E"])
        df.plot.box()
    Using 2 columns for clarity.
    """
    np.random.seed(123456)
    df = pd.DataFrame(
        np.random.rand(10, 2), columns=["X", "Y"]
    )
    df.plot.box(ax=ax)
    return [
        _bxp_expected(df["X"].values, position=1),
        _bxp_expected(df["Y"].values, position=2),
    ]


def case_five_columns_from_docs(ax):
    """
    Five-column example from the pandas docs (positions 1–5).
    """
    np.random.seed(123456)
    df = pd.DataFrame(
        np.random.rand(10, 5), columns=["A", "B", "C", "D", "E"]
    )
    df.plot.box(ax=ax)
    return [
        _bxp_expected(df[col].values, position=i + 1)
        for i, col in enumerate(df.columns)
    ]


def case_custom_whis(ax):
    """Box with whis=2 — different whisker extent."""
    np.random.seed(7)
    df = pd.DataFrame({"P": np.random.randn(30), "Q": np.random.randn(30)})
    df.plot.box(whis=2.0, ax=ax)
    return [
        _bxp_expected(df["P"].values, position=1, whis=2.0),
        _bxp_expected(df["Q"].values, position=2, whis=2.0),
    ]
