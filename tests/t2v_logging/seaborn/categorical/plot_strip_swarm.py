"""
Test cases for ``sns.stripplot``, ``sns.swarmplot`` and
``catplot(kind="strip"|"swarm")``.

How it is logged
----------------
Both strip and swarm use ``_CategoricalPlotter.plot_strips`` /
``plot_swarms``, which are patched by ``activate_sns_categorical_scatter_plots``.
The logged plt_func is ``"scatter"`` with one series per category group.

Logged structure
----------------
- **x** = category integer index (0, 1, 2, …) repeated for every point in
  that group — same for all points in a group (no jitter in logged coords).
- **y** = data values for that group, in seaborn's internal processing order
  (may differ from the DataFrame row order).

Strategy
--------
- **x values** are fully computable: categories are ordered alphabetically
  by seaborn, so ``sorted(df[x_col].unique())`` gives the index mapping.
- **y values** are NOT checked (internal order is unknown).
- ``UnorderedSeries`` is used since catplot may draw groups in any order.
- **Axes-level** functions produce one extra empty (n=0) "scatter" artefact.

Figure-level (catplot)
    no artefact → n_series = n_categories

Axes-level (stripplot / swarmplot)
    artefact    → n_series = n_categories + 1
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.categorical._datasets import make_tips


def _strip_expected(df, x_col, y_col, has_artefact: bool) -> list:
    """
    Build expected series for a strip/swarm plot.

    x is verified (constant integer position per group).
    y is skipped (seaborn processing order is undefined).
    """
    categories = list(df[x_col].unique())   # first-appearance order (seaborn default)
    expected   = UnorderedSeries()
    for i, cat in enumerate(categories):
        n = int((df[x_col] == cat).sum())
        expected.append({
            "plt_func": "scatter",
            "x": [float(i)] * n,    # position is constant per category
            # y: not checked — internal order unknown
        })
    if has_artefact:
        expected.append({"plt_func": "scatter"})   # empty legend handle
    return expected


# ---------------------------------------------------------------------------
# catplot (figure-level) — no artefact
# ---------------------------------------------------------------------------

def case_catplot_strip_basic(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill")
    → 1 scatter series per day (4 total).
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill")
    return _strip_expected(tips, "day", "total_bill", has_artefact=False)


def case_catplot_strip_hue(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="swarm")

    Hue changes point colours only — seaborn still produces one scatter series
    per x-category (not one per day×sex combination).
    → Same structure as case_catplot_strip_basic (n_days series).
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill", hue="sex", kind="swarm")
    return _strip_expected(tips, "day", "total_bill", has_artefact=False)


def case_catplot_swarm_basic(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill", kind="swarm")
    → Same structure as strip.
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill", kind="swarm")
    return _strip_expected(tips, "day", "total_bill", has_artefact=False)


# ---------------------------------------------------------------------------
# Axes-level — one extra empty artefact
# ---------------------------------------------------------------------------

def case_stripplot_basic(ax):
    """
    Axes-level strip plot.
    Tutorial: sns.swarmplot(data=tips, x="day", y="total_bill")
    → [empty artefact, 1 scatter per day].
    """
    tips = make_tips()
    sns.stripplot(data=tips, x="day", y="total_bill", ax=ax)
    return _strip_expected(tips, "day", "total_bill", has_artefact=True)


def case_swarmplot_basic(ax):
    """
    Axes-level swarm plot.
    """
    tips = make_tips()
    sns.swarmplot(data=tips, x="day", y="total_bill", ax=ax)
    return _strip_expected(tips, "day", "total_bill", has_artefact=True)
