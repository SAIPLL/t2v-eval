"""
Test cases for ``sns.catplot`` features covered in the tutorial:
faceting (col/row), orient, order, and kind="boxen".

All are figure-level (no empty-series artefact).

Kind mapping recap
------------------
    "strip"  / "swarm"  → "scatter" per x-category
    "box"    / "violin" → "bxp"     per x-category
    "boxen"             → "bxp"     per x-category (letter-value plot)
    "bar"               → "bar" + "plot" (CI) per hue level
    "count"             → "bar" (one series, all categories)
    "point"             → "plot" per hue level

Faceted catplot
---------------
``col="time"`` or ``row="class"`` splits data into subsets, one per facet.
Each facet produces its own set of series.  The total series count equals
the per-facet count × number of facets.
"""

import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.categorical._datasets import (
    make_tips, make_titanic,
)
from tests.t2v_logging.seaborn.categorical.plot_box_violin import _bxp_expected


# ---------------------------------------------------------------------------
# order= and orient= options
# ---------------------------------------------------------------------------

def case_catplot_order(ax):
    """
    Tutorial: sns.catplot(data=tips, x="smoker", y="tip", order=["No","Yes"])

    Explicit order is respected.  Strip plot, 2 categories.
    """
    tips = make_tips()
    sns.catplot(data=tips, x="smoker", y="tip", order=["No", "Yes"])
    # order param controls the category positions
    order = ["No", "Yes"]
    expected = UnorderedSeries()
    for i, cat in enumerate(order):
        n = int((tips["smoker"] == cat).sum())
        expected.append({"plt_func": "scatter", "x": [float(i)] * n})
    return expected


def case_catplot_orient_y(ax):
    """
    Tutorial: sns.catplot(data=tips, x="total_bill", y="day", hue="time", kind="swarm")

    Horizontal orientation (y is the categorical axis).
    Scatter per y-category — same structure, orient swaps x/y roles.
    """
    tips = make_tips()
    sns.catplot(data=tips, x="total_bill", y="day", hue="time", kind="swarm")
    n_cats = tips["day"].nunique()
    return [{"plt_func": "scatter"}] * n_cats


# ---------------------------------------------------------------------------
# kind="boxen" (letter-value plot)
# ---------------------------------------------------------------------------

def case_catplot_boxen(ax):
    """
    Tutorial: sns.catplot(data=diamonds, x="color", y="price", kind="boxen")

    Logged as bxp series (same patch as box/violin).
    Partial check — letter-value whisker extent differs from IQR rule.
    """
    import numpy as np
    rng = __import__("numpy").random.default_rng(7)
    diamonds = __import__("pandas").DataFrame({
        "color": rng.choice(list("DEFGHIJ"), 500),
        "price": rng.integers(300, 18000, 500).astype(float),
    })
    sns.catplot(data=diamonds.sort_values("color"),
                x="color", y="price", kind="boxen")
    n_cats = diamonds["color"].nunique()
    return [{"plt_func": "bxp"}] * n_cats


# ---------------------------------------------------------------------------
# Faceted plots
# ---------------------------------------------------------------------------

def case_catplot_col_facet_swarm(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill",
                          hue="smoker", kind="swarm", col="time")

    col="time" → 2 facets (Lunch, Dinner), each with n_days scatter series.
    Total series = n_time_vals × n_days.
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill",
                hue="smoker", kind="swarm", col="time", aspect=0.7)
    n_facets = tips["time"].nunique()
    n_cats   = tips["day"].nunique()
    return [{"plt_func": "scatter"}] * (n_facets * n_cats)


def case_catplot_row_facet_box(ax):
    """
    Tutorial:
        g = sns.catplot(data=titanic, x="fare", y="embark_town",
                        row="class", kind="box", orient="h", ...)

    row="class" → 3 facets (First/Second/Third class).
    Each facet: box plot of fare per embark_town (horizontal).
    Total series = n_class × n_embark_town.
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, x="fare", y="embark_town",
                row="class", kind="box", orient="h",
                sharex=False, height=1.5, aspect=4)
    n_facets = titanic["class"].nunique()
    n_cats   = titanic["embark_town"].nunique()
    return [{"plt_func": "bxp"}] * (n_facets * n_cats)


# ---------------------------------------------------------------------------
# Violin with split / inner options
# ---------------------------------------------------------------------------

def case_catplot_violin_split(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill",
                          hue="sex", kind="violin", split=True)

    split=True → still logged as bxp per (day × sex) group.
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill",
                hue="sex", kind="violin", split=True)
    n = tips["day"].nunique() * tips["sex"].nunique()
    return [{"plt_func": "bxp"}] * n


# ---------------------------------------------------------------------------
# Count with hue and facet
# ---------------------------------------------------------------------------

def case_catplot_count_hue(ax):
    """
    Tutorial: sns.catplot(data=titanic, y="deck", hue="class", kind="count")

    y="deck" makes this horizontal → logged as "barh" per hue level.
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, y="deck", hue="class",
                kind="count", palette="pastel", edgecolor=".6")
    n_hue = titanic["class"].nunique()
    return [{"plt_func": "barh"}] * n_hue


# ---------------------------------------------------------------------------
# Point with hue
# ---------------------------------------------------------------------------

def case_catplot_point_hue(ax):
    """
    Tutorial: sns.catplot(data=titanic, x="class", y="survived",
                          hue="sex", kind="point")

    Each hue level produces: 1 connecting line (n=n_x_cats) + n_x_cats CI
    plots (n=2 each, one per x-category).
    Total = n_hue × (1 + n_x_cats) series.
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, x="class", y="survived",
                hue="sex", kind="point")
    n_hue   = titanic["sex"].nunique()
    n_x     = titanic["class"].nunique()
    return [{"plt_func": "plot"}] * (n_hue * (1 + n_x))
