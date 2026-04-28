"""
Test cases for ``sns.residplot`` (residual scatter plot).

How it is logged
----------------
``residplot`` draws:

    ax.axhline(0, …)         → plt_func = "axhline"  (zero reference line)
    ax.scatter(x, resid, …)  → plt_func = "scatter"   (residual points)

Plus the axes-level empty "plot" artefact.

Logged series order
-------------------
    [0] "axhline"   n=2   ← zero line (two endpoints)
    [1] "plot"      n=0   ← empty legend artefact
    [2] "scatter"   n=data ← residual points

Strategy
--------
- **axhline**: partial check (plt_func only; x range is matplotlib-internal).
- **plot** artefact: partial check.
- **scatter** residuals: residuals = y - ŷ, depend on the fitted model
  → partial check (plt_func + length consistency).
"""

import seaborn as sns

from tests.t2v_logging.seaborn.regression._datasets import (
    make_anscombe, make_tips,
)


def case_linear_data(ax):
    """
    Tutorial: sns.residplot(x="x", y="y", data=anscombe.query("dataset=='I'"))
    Residuals from a linear fit on Anscombe dataset I.
    """
    ds = make_anscombe()["I"]
    sns.residplot(x="x", y="y", data=ds, scatter_kws={"s": 80}, ax=ax)
    return [
        {"plt_func": "axhline"},    # zero reference line
        {"plt_func": "plot"},       # empty artefact
        {"plt_func": "scatter"},    # residual points (n=11, partial)
    ]


def case_nonlinear_data(ax):
    """
    Tutorial: sns.residplot on Anscombe II (quadratic pattern in residuals).
    Same series structure — linear model mis-fit shows up in residual shape.
    """
    ds = make_anscombe()["II"]
    sns.residplot(x="x", y="y", data=ds, scatter_kws={"s": 80}, ax=ax)
    return [
        {"plt_func": "axhline"},
        {"plt_func": "plot"},
        {"plt_func": "scatter"},
    ]


def case_tips(ax):
    """
    Tutorial: sns.residplot(x="total_bill", y="tip", data=tips)
    """
    tips = make_tips()
    sns.residplot(x="total_bill", y="tip", data=tips, ax=ax)
    return [
        {"plt_func": "axhline"},
        {"plt_func": "plot"},
        {"plt_func": "scatter"},
    ]
