"""
Test cases for ``sns.barplot``, ``sns.countplot``, ``sns.pointplot`` and
their ``catplot`` equivalents.

How each is logged
------------------
**barplot / catplot kind="bar"**
    Seaborn draws bars via ``ax.bar()`` and CI lines via ``ax.plot()``.
    Without hue (axes-level): [legend artefact "bar", data "bar", CI "plot" ×2]
    With hue / catplot: multiple bar + plot series per group.
    → Partial check (plt_func + count).

**countplot / catplot kind="count"**
    Seaborn draws one bar per category; x = integer position, y = count.
    catplot: 1 "bar" series with all categories.
    axes-level: [legend artefact (nan values), data "bar"].
    → Computed (value_counts → category positions).

**pointplot / catplot kind="point"**
    Seaborn draws points + connecting lines + error bars via ``ax.plot()``.
    → Partial check (plt_func + count).
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.categorical._datasets import make_tips, make_titanic


# ---------------------------------------------------------------------------
# countplot helpers
# ---------------------------------------------------------------------------

def _count_expected(df, col) -> dict:
    """
    Compute expected count bar series.

    Categories ordered alphabetically (seaborn default for strings).
    x = integer positions 0, 1, …, n-1.
    y = count per category in alphabetical order.
    """
    cats   = list(df[col].unique())   # first-appearance order
    counts = [float((df[col] == cat).sum()) for cat in cats]
    return {"plt_func": "bar",
            "x": list(range(len(cats))),
            "y": counts}


# ---------------------------------------------------------------------------
# barplot cases
# ---------------------------------------------------------------------------

def case_barplot_basic(ax):
    """
    Tutorial: sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")

    bar = mean estimate, plot = CI lines.
    Partial check: correct plt_func mix and count.
    """
    titanic = make_titanic()
    sns.barplot(data=titanic, x="sex", y="survived", ax=ax)
    # axes-level without hue: [legend artefact, bar (means), plot CI ×2]
    return [
        {"plt_func": "bar"},    # legend artefact (nan)
        {"plt_func": "bar"},    # mean bars
        {"plt_func": "plot"},   # CI lower
        {"plt_func": "plot"},   # CI upper
    ]


def case_catplot_bar_hue(ax):
    """
    Tutorial: sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")

    Seaborn draws one bar series per hue level (containing all x positions),
    plus 2 CI plot series per hue level.
    → n_hue_levels × (1 bar + 2 CI plots) = n_class × 3 series total.
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="bar")
    n_hue = titanic["class"].nunique()
    return [{"plt_func": "bar"}, {"plt_func": "plot"}, {"plt_func": "plot"}] * n_hue


# ---------------------------------------------------------------------------
# countplot cases
# ---------------------------------------------------------------------------

def case_countplot_basic(ax):
    """
    Tutorial: sns.catplot(data=titanic, x="deck", kind="count")  (figure-level)

    → 1 bar series with x = [0,1,2,3,4], y = count per deck.
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, x="deck", kind="count")
    return [_count_expected(titanic, "deck")]


def case_countplot_axeslevel(ax):
    """
    Axes-level countplot.
    → [legend artefact (nan), bar series with counts].
    """
    tips = make_tips()
    sns.countplot(data=tips, x="day", ax=ax)
    return [
        {"plt_func": "bar"},            # legend artefact (nan values)
        _count_expected(tips, "day"),   # actual counts
    ]


# ---------------------------------------------------------------------------
# pointplot cases
# ---------------------------------------------------------------------------

def case_pointplot_basic(ax):
    """
    Tutorial: sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="point")

    Points + CI + connecting lines all drawn via ax.plot().
    Partial check.
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, x="sex", y="survived", hue="class", kind="point")
    # 1 plot series per hue level (connects points) + 2 CI plot per level
    n_hue = titanic["class"].nunique()
    return [{"plt_func": "plot"}] * (n_hue * 3)


def case_catplot_point_no_hue(ax):
    """
    Point plot without hue.
    → 1 connecting line + 2 CI plots (partial).
    """
    titanic = make_titanic()
    sns.catplot(data=titanic, x="sex", y="survived", kind="point")
    return [
        {"plt_func": "plot"},   # connecting line / dots
        {"plt_func": "plot"},   # CI lower caps
        {"plt_func": "plot"},   # CI upper caps
    ]
