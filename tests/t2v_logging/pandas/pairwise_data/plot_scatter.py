"""
Test cases for ``DataFrame.plot.scatter`` logging.

Strategy: computed
------------------
Delegates to ``ax.scatter()``.  Logged values::

    plt_func = "scatter"
    x        = df[x_col].values
    y        = df[y_col].values

One series per call.
"""

import numpy as np
import pandas as pd


def case_basic(ax):
    """Simplest scatter — two numeric columns."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    df.plot.scatter(x="a", y="b", ax=ax)
    return [{"plt_func": "scatter",
             "x": df["a"].values, "y": df["b"].values}]


def case_with_size(ax):
    """Scatter with a per-point size column — x, y, and s are all checked."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 1.0, 4.0, 3.0],
        "s": [10.0, 50.0, 30.0, 80.0],
    })
    df.plot.scatter(x="a", y="b", s=df["s"], ax=ax)
    return [{"plt_func": "scatter",
             "x": df["a"].values,
             "y": df["b"].values,
             "s": df["s"].values}]


def case_two_groups(ax):
    """Two scatter calls on the same axes — two logged series."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0],
        "c": [7.0, 8.0, 9.0], "d": [1.0, 2.0, 3.0],
    })
    df.plot.scatter(x="a", y="b", ax=ax)
    df.plot.scatter(x="c", y="d", ax=ax)
    return [
        {"plt_func": "scatter", "x": df["a"].values, "y": df["b"].values},
        {"plt_func": "scatter", "x": df["c"].values, "y": df["d"].values},
    ]
