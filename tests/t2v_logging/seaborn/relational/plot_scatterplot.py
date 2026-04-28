"""
Test cases for ``sns.scatterplot`` / ``sns.relplot(kind="scatter")`` logging.

How it is logged
----------------
The seaborn patch replaces ``_ScatterPlotter.plot`` entirely.  At the end of
that method, it calls ``pathcollection_to_logdata(points)`` and logs one
``"scatter"`` series containing all data points.

Empty-series artefact
---------------------
When **no** hue semantic is used, seaborn creates an empty PathCollection
as a legend handle.  This fires our matplotlib scatter patch and produces an
extra series with ``n=0``.  The expected list therefore starts with a
partial-check entry ``{"plt_func": "scatter"}`` for this artefact.

When a **hue** semantic is used, seaborn handles legend creation differently
and no empty series appears.

Logged structure
----------------
    no hue:   [{"plt_func":"scatter", n=0}, {"plt_func":"scatter", x=total_bill, y=tip}]
    with hue: [{"plt_func":"scatter", x=total_bill, y=tip}]
"""

import seaborn as sns

from tests.t2v_logging.seaborn.relational._datasets import make_tips


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scatter_expected(df, x_col, y_col, has_hue: bool) -> list:
    """Build expected series list for a scatter plot."""
    data_series = {
        "plt_func": "scatter",
        "x": df[x_col].values,
        "y": df[y_col].values,
    }
    if has_hue:
        return [data_series]
    # Without hue: empty legend-handle series precedes the data series
    return [{"plt_func": "scatter"}, data_series]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def case_basic(ax):
    """
    Basic scatter — no semantics.
    Tutorial: sns.relplot(data=tips, x="total_bill", y="tip")
    """
    tips = make_tips()
    sns.scatterplot(data=tips, x="total_bill", y="tip", ax=ax)
    return _scatter_expected(tips, "total_bill", "tip", has_hue=False)


def case_hue_categorical(ax):
    """
    Scatter with a categorical hue — one series for all points.
    Tutorial: sns.relplot(..., hue="smoker")
    """
    tips = make_tips()
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker", ax=ax)
    return _scatter_expected(tips, "total_bill", "tip", has_hue=True)


def case_hue_and_style(ax):
    """
    Scatter with hue + style semantics — same single series, different markers.
    Tutorial: sns.relplot(..., hue="smoker", style="smoker")
    """
    tips = make_tips()
    sns.scatterplot(
        data=tips, x="total_bill", y="tip",
        hue="smoker", style="smoker", ax=ax,
    )
    return _scatter_expected(tips, "total_bill", "tip", has_hue=True)


def case_size_semantic(ax):
    """
    Scatter with a size semantic (numeric column).
    Tutorial: sns.relplot(..., size="size")

    Seaborn normalises the raw ``size`` column to area units, so ``s``
    cannot be computed from the input.  Expected sizes are extracted from
    ``ax.collections[-1].get_sizes()`` — the same source the logging patch
    reads.
    """
    tips = make_tips()
    sns.scatterplot(data=tips, x="total_bill", y="tip", size="size", ax=ax)
    ax.figure.canvas.draw()
    s_vals = ax.collections[-1].get_sizes()
    # Without hue: empty artefact + full data series
    return [
        {"plt_func": "scatter"},
        {"plt_func": "scatter",
         "x": tips["total_bill"].values,
         "y": tips["tip"].values,
         "s": s_vals},
    ]
