"""
Test cases for ``sns.boxplot``, ``sns.violinplot`` and
``catplot(kind="box"|"violin")``.

How it is logged
----------------
Both box and violin use ``_CategoricalPlotter.plot_boxes`` /
``plot_violins``, which log the 5-number summary as ``"bxp"`` series.

Logged structure (one series per category group)
-------------------------------------------------
    plt_func = "bxp"
    x        = [category_position] × 5   (0-indexed: 0, 1, 2, …)
    y        = sorted [whislo, Q1, median, Q3, whishi]

Category positions: seaborn orders string columns alphabetically by default,
so ``sorted(df[col].unique())`` gives the index mapping
(0 → first alphabetical, 1 → second, …).

Figure-level (catplot)
    no artefact → n_series = n_categories (or n_categories × n_hue for hue)

Axes-level (boxplot / violinplot)
    1 empty "fill_between" artefact → n_series = n_categories + 1
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries
from tests.t2v_logging.seaborn.categorical._datasets import make_tips


def _bxp_expected(data: np.ndarray, position: float, whis: float = 1.5) -> dict:
    """Compute expected bxp series for one categorical group."""
    arr = data[~np.isnan(data)]
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr    = q3 - q1
    whislo = arr[arr >= q1 - whis * iqr].min()
    whishi = arr[arr <= q3 + whis * iqr].max()
    y = sorted([float(whislo), float(q1), float(med), float(q3), float(whishi)])
    return {"plt_func": "bxp", "x": [float(position)] * 5, "y": y}


def _box_expected(df, x_col, y_col, has_artefact: bool,
                  artefact_func: str = "fill_between") -> list:
    """
    Build expected series for a box/violin plot (no hue).

    ``UnorderedSeries`` used since catplot may draw categories in any order.
    """
    categories = list(df[x_col].unique())   # first-appearance order
    expected   = UnorderedSeries()
    for i, cat in enumerate(categories):
        data = df.loc[df[x_col] == cat, y_col].values
        expected.append(_bxp_expected(data, position=float(i)))
    if has_artefact:
        expected.append({"plt_func": artefact_func})
    return expected


# ---------------------------------------------------------------------------
# catplot (figure-level)
# ---------------------------------------------------------------------------

def case_catplot_box_basic(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill", kind="box")
    → 1 bxp series per day (4 total), positions 0–3.
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill", kind="box")
    return _box_expected(tips, "day", "total_bill", has_artefact=False)


def case_catplot_box_hue(ax):
    """
    Tutorial: sns.catplot(data=tips, x="day", y="total_bill", hue="smoker", kind="box")
    → 1 bxp per (day × smoker) = 4 × 2 = 8 series (partial, hue order unknown).
    """
    tips = make_tips()
    sns.catplot(data=tips, x="day", y="total_bill", hue="smoker", kind="box")
    n = tips["day"].nunique() * tips["smoker"].nunique()
    return [{"plt_func": "bxp"}] * n


def case_catplot_violin_basic(ax):
    """
    Tutorial: sns.catplot(data=tips, x="total_bill", y="day", hue="sex", kind="violin")
    → 1 bxp per (day × sex) = 4 × 2 = 8 series (partial — hue order unknown).
    """
    tips = make_tips()
    sns.catplot(data=tips, x="total_bill", y="day", hue="sex", kind="violin")
    n = tips["day"].nunique() * tips["sex"].nunique()
    return [{"plt_func": "bxp"}] * n


# ---------------------------------------------------------------------------
# Axes-level — 1 empty fill_between artefact before the data series
# ---------------------------------------------------------------------------

def case_boxplot_basic(ax):
    """
    Tutorial: sns.boxplot(data=tips, x="day", y="total_bill")
    → [empty fill_between, bxp × n_days].
    """
    tips = make_tips()
    sns.boxplot(data=tips, x="day", y="total_bill", ax=ax)
    return _box_expected(tips, "day", "total_bill", has_artefact=True,
                         artefact_func="fill_between")


def case_violinplot_basic(ax):
    """
    Tutorial: sns.violinplot(data=tips, x="day", y="total_bill")
    → [empty fill_between, bxp × n_days].
    """
    tips = make_tips()
    sns.violinplot(data=tips, x="day", y="total_bill", ax=ax)
    return _box_expected(tips, "day", "total_bill", has_artefact=True,
                         artefact_func="fill_between")
