"""
Test cases for ``sns.lmplot`` (figure-level regression plot).

How it is logged
----------------
``lmplot`` wraps ``regplot`` inside a ``FacetGrid``.
It produces **no empty-series artefact** (figure-level behaviour).

Per-facet series pattern
------------------------
    With CI    : scatter  + plot  + fill_between
    ci=None    : scatter  + plot
    lowess/log : scatter  + plot  [+ fill_between if CI requested]

With hue (2 levels, one facet)
    n_hue × (scatter + plot + fill_between)  = 6 series

With col="time", hue="smoker" (2 cols × 2 hue)
    n_col × n_hue × (scatter + plot + fill_between)  = 12 series

Strategy
--------
- **scatter** matches input data → x/y values checked.
- **plot** (line) and **fill_between** (CI) → partial checks.
- Hue / facet cases use ``UnorderedSeries`` since seaborn controls draw order.
"""

import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.regression._datasets import (
    make_anscombe, make_tips,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _regplot_group(scatter_x, scatter_y, with_ci: bool = True) -> list:
    """One regression group: scatter (values checked) + line + optional CI."""
    group = [
        {"plt_func": "scatter", "x": scatter_x, "y": scatter_y},
        {"plt_func": "plot"},   # regression line (partial)
    ]
    if with_ci:
        group.append({"plt_func": "fill_between"})   # CI band (partial)
    return group


# ---------------------------------------------------------------------------
# Single-facet cases
# ---------------------------------------------------------------------------

def case_basic(ax):
    """
    Tutorial: sns.lmplot(x="total_bill", y="tip", data=tips)
    Default 95% CI → 3 series.
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="tip", data=tips)
    return _regplot_group(tips["total_bill"].values, tips["tip"].values)


def case_ci_none(ax):
    """
    Tutorial: sns.lmplot(…, ci=None)
    No CI → 2 series.
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="tip", data=tips, ci=None)
    return _regplot_group(tips["total_bill"].values, tips["tip"].values,
                          with_ci=False)


def case_logistic(ax):
    """
    Tutorial: sns.lmplot(x="total_bill", y="big_tip", data=tips, logistic=True)

    ``y_jitter=.03`` adds random jitter to y before the scatter call, so the
    logged y values differ from the original column.  Only x is checked.
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="big_tip", data=tips,
               logistic=True, y_jitter=.03)
    return [
        {"plt_func": "scatter", "x": tips["total_bill"].values},  # y jittered
        {"plt_func": "plot"},           # logistic curve
        {"plt_func": "fill_between"},   # CI band
    ]


def case_lowess(ax):
    """
    Tutorial: sns.lmplot(…, lowess=True)
    LOWESS smoother — no CI → 2 series.
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="tip", data=tips, lowess=True)
    return _regplot_group(tips["total_bill"].values, tips["tip"].values,
                          with_ci=False)


def case_polynomial_order2(ax):
    """
    Tutorial: sns.lmplot(x="x", y="y", data=anscombe_II, order=2, ci=None)
    Polynomial fit — no CI → 2 series.
    """
    ds = make_anscombe()["II"]
    sns.lmplot(x="x", y="y", data=ds, order=2, ci=None,
               scatter_kws={"s": 80})
    return _regplot_group(ds["x"].values, ds["y"].values, with_ci=False)


def case_robust(ax):
    """
    Tutorial: sns.lmplot(…, robust=True, ci=None)
    Robust regression — no CI → 2 series.
    """
    ds = make_anscombe()["III"]
    sns.lmplot(x="x", y="y", data=ds, robust=True, ci=None,
               scatter_kws={"s": 80})
    return _regplot_group(ds["x"].values, ds["y"].values, with_ci=False)


# ---------------------------------------------------------------------------
# Multi-group / faceted cases
# ---------------------------------------------------------------------------

def case_hue(ax):
    """
    Tutorial: sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)
    2 hue levels → 6 series (2 × scatter+plot+fill_between).

    A plain partial list is used (not UnorderedSeries) because the expected
    list mixes scatter series (with x/y) and partial plot/fill_between dicts
    (no x/y).  Partial dicts all sort to the same key, breaking order-independent
    matching.  The count and plt_func of all 6 series are still verified.
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)
    n_hue = tips["smoker"].nunique()
    return [{"plt_func": "scatter"},
            {"plt_func": "plot"},
            {"plt_func": "fill_between"}] * n_hue


def case_col_hue(ax):
    """
    Tutorial: sns.lmplot(x="total_bill", y="tip", hue="smoker",
                          col="time", data=tips)
    2 cols × 2 hue → 12 series.
    Partial check — facet × hue ordering is seaborn-controlled.
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="tip", hue="smoker", col="time", data=tips)
    n_col = tips["time"].nunique()
    n_hue = tips["smoker"].nunique()
    # Each (col, hue) group: scatter + plot + fill_between = 3 series
    return [{"plt_func": "scatter"},
            {"plt_func": "plot"},
            {"plt_func": "fill_between"}] * (n_col * n_hue)


def case_row_col_hue(ax):
    """
    Tutorial: sns.lmplot(…, col="time", row="sex", data=tips, height=3)
    2 rows × 2 cols × 2 hue → 24 series (partial).
    """
    tips = make_tips()
    sns.lmplot(x="total_bill", y="tip", hue="smoker",
               col="time", row="sex", data=tips, height=3)
    n = tips["time"].nunique() * tips["sex"].nunique() * tips["smoker"].nunique()
    return [{"plt_func": "scatter"},
            {"plt_func": "plot"},
            {"plt_func": "fill_between"}] * n
