"""
Test cases for ``sns.regplot`` (axes-level regression plot).

How it is logged
----------------
``regplot`` calls three matplotlib functions directly, each of which is
patched by T2V logging:

    ax.scatter(x, y, ...)    → plt_func = "scatter"   (data points)
    ax.plot(x_fit, y_fit, …) → plt_func = "plot"      (regression line)
    ax.fill_between(…)       → plt_func = "fill_between" (95% CI band)

Logged series order (with CI)
------------------------------
    [0] "plot"         n=0     ← empty legend artefact (axes-level)
    [1] "scatter"      n=data  ← actual scatter points  (values checked)
    [2] "plot"         n=100   ← regression line on 100-pt grid (partial)
    [3] "fill_between" n=203   ← CI band  (partial, complex shape)

Without CI (``ci=None``)
    [0] "plot"         n=0     ← empty artefact
    [1] "scatter"      n=data
    [2] "plot"         n=100   ← regression line

Strategy
--------
- **scatter** x and y match the input DataFrame columns → computed.
- **plot** (regression line) and **fill_between** (CI) depend on seaborn's
  internal scipy regression → partial check (plt_func + length only).
"""

import seaborn as sns

from tests.t2v_logging.seaborn.regression._datasets import (
    make_anscombe, make_tips,
)


def case_basic_with_ci(ax):
    """
    Tutorial: sns.regplot(x="total_bill", y="tip", data=tips)
    Default 95% CI → 4 series.
    """
    tips = make_tips()
    sns.regplot(x="total_bill", y="tip", data=tips, ax=ax)
    return [
        {"plt_func": "plot"},           # empty legend artefact
        {"plt_func": "scatter",
         "x": tips["total_bill"].values,
         "y": tips["tip"].values},      # data points — values checked
        {"plt_func": "plot"},           # regression line (partial)
        {"plt_func": "fill_between"},   # CI band (partial)
    ]


def case_ci_none(ax):
    """
    Tutorial: sns.regplot(…, ci=None)
    No CI band → 3 series.
    """
    tips = make_tips()
    sns.regplot(x="total_bill", y="tip", data=tips, ci=None, ax=ax)
    return [
        {"plt_func": "plot"},           # empty artefact
        {"plt_func": "scatter",
         "x": tips["total_bill"].values,
         "y": tips["tip"].values},
        {"plt_func": "plot"},           # regression line
    ]


def case_anscombe_linear(ax):
    """
    Tutorial: sns.regplot on Anscombe dataset I (linear relationship).
    """
    ds = make_anscombe()["I"]
    sns.regplot(x="x", y="y", data=ds, ci=None,
                scatter_kws={"s": 80}, ax=ax)
    return [
        {"plt_func": "plot"},
        {"plt_func": "scatter", "x": ds["x"].values, "y": ds["y"].values},
        {"plt_func": "plot"},
    ]


def case_anscombe_polynomial(ax):
    """
    Tutorial: sns.regplot(…, order=2, ci=None)
    Polynomial (quadratic) fit — same series structure as linear.
    """
    ds = make_anscombe()["II"]
    sns.regplot(x="x", y="y", data=ds, order=2, ci=None,
                scatter_kws={"s": 80}, ax=ax)
    return [
        {"plt_func": "plot"},
        {"plt_func": "scatter", "x": ds["x"].values, "y": ds["y"].values},
        {"plt_func": "plot"},
    ]
