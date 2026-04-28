"""
Test cases for ``DataFrame.plot.area`` logging.

Strategy: computed (uses fill_between_expected)
------------------------------------------------
Each column produces **two** logged series in this order::

    [2k]   plt_func="plot"        — the upper edge line  (ax.plot)
    [2k+1] plt_func="fill_between" — the filled region   (ax.fill_between)

For stacked=True (default), the cumulative stack is:
    stack_0 = col_0
    stack_k = stack_{k-1} + col_k

Each fill_between call: fill_between(x, y1=stack_{k-1}, y2=stack_k)
where stack_{-1} = 0 (zero baseline).

The same fill_between_expected helper from helpers.py is used.
"""

import numpy as np
import pandas as pd

from tests.t2v_logging.helpers import fill_between_expected


def case_stacked_two_columns(ax):
    """
    Two-column stacked area — 4 logged series total (plot+fill per column).

    Stacking: col_a first, then col_b on top.
    """
    df = pd.DataFrame({
        "a": [1.0, 2.0, 1.5],
        "b": [0.5, 1.0, 0.8],
    })
    df.plot.area(ax=ax)

    x      = df.index.values.astype(float)
    col_a  = df["a"].values
    col_b  = df["b"].values
    stack0 = col_a
    stack1 = col_a + col_b
    zeros  = np.zeros_like(x)

    return [
        # Column 'a': upper edge line + fill from 0 to col_a
        {"plt_func": "plot",         "x": x, "y": stack0},
        fill_between_expected(x, zeros,  stack0),
        # Column 'b': stacked upper edge + fill from col_a to col_a+col_b
        {"plt_func": "plot",         "x": x, "y": stack1},
        fill_between_expected(x, stack0, stack1),
    ]


def case_unstacked(ax):
    """
    Unstacked area (stacked=False) — each column fills from 0 independently.
    Still produces plot + fill_between per column.
    """
    df = pd.DataFrame({
        "a": [1.0, 3.0, 2.0],
        "b": [2.0, 1.0, 3.0],
    })
    df.plot.area(stacked=False, ax=ax)

    x     = df.index.values.astype(float)
    zeros = np.zeros_like(x)

    return [
        {"plt_func": "plot",          "x": x, "y": df["a"].values},
        fill_between_expected(x, zeros, df["a"].values),
        {"plt_func": "plot",          "x": x, "y": df["b"].values},
        fill_between_expected(x, zeros, df["b"].values),
    ]
