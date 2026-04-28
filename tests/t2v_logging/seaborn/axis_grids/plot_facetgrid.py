"""
Test cases for ``sns.FacetGrid`` with various mapped functions.

Key observation: no-artefact rule
----------------------------------
``FacetGrid.map()`` calls through seaborn's internal plotter, NOT through
the axes-level function's artefact-generating code path.  Result: zero
empty-series artefacts regardless of mapped function.

Series count formula
--------------------
    n_series = n_col_facets × n_row_facets × n_hue_groups × n_per_cell

where ``n_per_cell`` depends on the mapped function:

    histplot        → 1 "bar" per cell
    scatterplot     → 1 "scatter" per hue group per cell
    regplot (no CI) → 1 "scatter" per cell  (fit_reg=False suppresses line+CI)
    barplot         → 1 "bar" + 2 "plot" (CI) = 3 per cell
    kdeplot         → 1 "plot" per cell
    plt.scatter     → 1 "scatter" per hue group per cell (matplotlib patch)
    pointplot(ci=None) → 1 "plot" per cell

All cases use partial checks (series count + plt_func only).
"""

import matplotlib.pyplot as plt
import seaborn as sns

from tests.t2v_logging.seaborn.axis_grids._datasets import (
    make_attend, make_tips,
)


def case_histplot_col(ax):
    """
    Tutorial: g = FacetGrid(tips, col="time"); g.map(histplot, "tip")
    2 time facets × 1 bar each = 2 "bar" total.
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, col="time")
    g.map(sns.histplot, "tip")
    n_facets = tips["time"].nunique()
    return [{"plt_func": "bar"}] * n_facets


def case_scatterplot_col_hue(ax):
    """
    Tutorial: g = FacetGrid(tips, col="sex", hue="smoker");
              g.map(scatterplot, "total_bill", "tip")
    2 col × 2 hue = 4 "scatter".
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, col="sex", hue="smoker")
    g.map(sns.scatterplot, "total_bill", "tip", alpha=.7)
    n = tips["sex"].nunique() * tips["smoker"].nunique()
    return [{"plt_func": "scatter"}] * n


def case_regplot_no_fit(ax):
    """
    Tutorial: g = FacetGrid(tips, row="smoker", col="time");
              g.map(regplot, "size", "total_bill", fit_reg=False)
    4 facets × 1 "scatter" each (fit_reg=False → no line, no CI, no artefact).
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
    g.map(sns.regplot, "size", "total_bill",
          color=".3", fit_reg=False, x_jitter=.1)
    n = tips["smoker"].nunique() * tips["time"].nunique()
    return [{"plt_func": "scatter"}] * n


def case_barplot_col(ax):
    """
    Tutorial: g = FacetGrid(tips, col="day"); g.map(barplot, "sex", "total_bill")
    4 day facets × (1 bar + 2 CI plot) = 12 series.
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
    g.map(sns.barplot, "sex", "total_bill", order=["Male", "Female"])
    n_facets = tips["day"].nunique()
    # barplot: 1 bar + 2 CI plot lines per facet
    return ([{"plt_func": "bar"}] +
            [{"plt_func": "plot"}] * 2) * n_facets


def case_kdeplot_row(ax):
    """
    Tutorial: g = FacetGrid(tips, row="day"); g.map(kdeplot, "total_bill")
    4 day facets × 1 KDE "plot" = 4 "plot".
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, row="day")
    g.map(sns.kdeplot, "total_bill")
    n_facets = tips["day"].nunique()
    return [{"plt_func": "plot"}] * n_facets


def case_plt_scatter_hue(ax):
    """
    Tutorial: g = FacetGrid(tips, hue="time"); g.map(plt.scatter, ...)
    1 merged facet with 2 hue groups × 1 "scatter" each = 2 "scatter".
    (plt.scatter → matplotlib ax.scatter → T2V scatter patch.)
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, hue="time", height=5)
    g.map(plt.scatter, "total_bill", "tip", s=100, alpha=.5)
    n_hue = tips["time"].nunique()
    return [{"plt_func": "scatter"}] * n_hue


def case_pointplot_col_wrap(ax):
    """
    Tutorial: g = FacetGrid(attend, col="subject", col_wrap=4);
              g.map(pointplot, "solutions", "score", errorbar=None)
    12 subject facets × 1 "plot" (no CI) = 12 "plot".
    """
    attend = make_attend()
    g      = sns.FacetGrid(attend, col="subject", col_wrap=4, height=2)
    g.map(sns.pointplot, "solutions", "score",
          order=[1, 2, 3], color=".3", errorbar=None)
    n_facets = attend["subject"].nunique()
    return [{"plt_func": "plot"}] * n_facets


def case_scatterplot_row_col(ax):
    """
    Tutorial: FacetGrid(tips, row="sex", col="smoker"); g.map(scatterplot, ...)
    2 row × 2 col = 4 facets × 1 "scatter" each = 4 "scatter".
    """
    tips = make_tips()
    g    = sns.FacetGrid(tips, row="sex", col="smoker",
                         margin_titles=True, height=2.5)
    g.map(sns.scatterplot, "total_bill", "tip", color="#334488")
    n = tips["sex"].nunique() * tips["smoker"].nunique()
    return [{"plt_func": "scatter"}] * n
