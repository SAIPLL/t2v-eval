"""
Test cases for ``sns.relplot`` (figure-level function).

Why a separate file
-------------------
``sns.relplot`` is a **figure-level** function — it creates its own
``FacetGrid`` and ignores any ``ax`` argument.  The case functions here
accept the ``ax`` parameter (to fit the shared test runner signature) but
do not use it; logging writes to the active ``log_dir`` regardless.

Key differences from axes-level (scatterplot / lineplot)
---------------------------------------------------------
- **No empty-series artefact** — ``relplot`` handles legend creation
  differently, so the empty PathCollection seen in the axes-level functions
  does not appear.
- **Faceted plots** — using ``col=`` or ``row=`` produces one series per
  facet (each subplot is a separate axes).

Logged structure summary
------------------------
    relplot scatter, no facet → 1 "scatter" series (all points)
    relplot scatter, col=X   → 1 "scatter" series per unique column value
    relplot line, errorbar=None, no hue → 1 "plot" series (mean line)
    relplot line, errorbar=None, hue=X → 1 "plot" series per hue level
    relplot line, default CI,   hue=X → ("plot" + "fill_between") per level
"""

import seaborn as sns

from tests.t2v_logging.seaborn.relational._datasets import make_fmri, make_tips


# ---------------------------------------------------------------------------
# Scatter cases
# ---------------------------------------------------------------------------

def case_scatter_basic(ax):
    """
    Basic scatter — direct counterpart of the tutorial's first example.
    Tutorial: sns.relplot(data=tips, x="total_bill", y="tip")

    → 1 scatter series, n = len(tips).
    """
    tips = make_tips()
    sns.relplot(data=tips, x="total_bill", y="tip")
    return [{"plt_func": "scatter",
             "x": tips["total_bill"].values,
             "y": tips["tip"].values}]


def case_scatter_hue(ax):
    """
    Scatter with categorical hue — still one series.
    Tutorial: sns.relplot(..., hue="smoker")

    → 1 scatter series, n = len(tips).
    """
    tips = make_tips()
    sns.relplot(data=tips, x="total_bill", y="tip", hue="smoker")
    return [{"plt_func": "scatter",
             "x": tips["total_bill"].values,
             "y": tips["tip"].values}]


def case_scatter_hue_style(ax):
    """
    Scatter with hue + style semantics.
    Tutorial: sns.relplot(..., hue="smoker", style="smoker")
    """
    tips = make_tips()
    sns.relplot(data=tips, x="total_bill", y="tip",
                hue="smoker", style="smoker")
    return [{"plt_func": "scatter",
             "x": tips["total_bill"].values,
             "y": tips["tip"].values}]


def case_scatter_size(ax):
    """
    Scatter with size semantic.
    Tutorial: sns.relplot(..., size="size")

    Raw ``size`` column values are normalised to area units by seaborn, so
    the logged ``s`` cannot be computed from the input directly.  Expected
    sizes are extracted from ``fg.ax.collections[-1].get_sizes()`` — the
    same source the logging patch reads.
    """
    tips = make_tips()
    fg = sns.relplot(data=tips, x="total_bill", y="tip", size="size")
    fg.figure.canvas.draw()
    s_vals = fg.ax.collections[-1].get_sizes()
    return [{"plt_func": "scatter",
             "x": tips["total_bill"].values,
             "y": tips["tip"].values,
             "s": s_vals}]


def case_scatter_facet_col(ax):
    """
    Faceted scatter — col="time" splits into one subplot per time value.
    Tutorial: sns.relplot(..., hue="smoker", col="time")

    → 1 scatter series per unique time value (2 total).
    Only plt_func and length are checked; values depend on the facet split.
    """
    tips = make_tips()
    sns.relplot(data=tips, x="total_bill", y="tip",
                hue="smoker", col="time")
    n_facets = tips["time"].nunique()
    return [{"plt_func": "scatter"}] * n_facets


# ---------------------------------------------------------------------------
# Line cases
# ---------------------------------------------------------------------------

def case_line_errorbar_none(ax):
    """
    Mean line, no CI, no hue.
    Tutorial: sns.relplot(..., kind="line", errorbar=None)

    → 1 plot series (mean per timepoint).
    """
    fmri = make_fmri()
    sns.relplot(data=fmri, x="timepoint", y="signal",
                kind="line", errorbar=None)
    means = fmri.groupby("timepoint")["signal"].mean()
    return [{"plt_func": "plot",
             "x": means.index.values.astype(float),
             "y": means.values}]


def case_line_hue_errorbar_none(ax):
    """
    One mean line per event, no CI.
    Tutorial: sns.relplot(..., kind="line", hue="event", errorbar=None)

    → 1 plot series per unique event (in order of first appearance).
    """
    fmri = make_fmri()
    sns.relplot(data=fmri, x="timepoint", y="signal",
                kind="line", hue="event", errorbar=None)
    events = list(fmri["event"].unique())
    expected = []
    for ev in events:
        m = fmri[fmri["event"] == ev].groupby("timepoint")["signal"].mean()
        expected.append({"plt_func": "plot",
                         "x": m.index.values.astype(float),
                         "y": m.values})
    return expected


def case_line_with_ci(ax):
    """
    Mean line + 95% CI, no hue.
    Tutorial: sns.relplot(..., kind="line")  ← default

    → 1 plot (mean) + 1 fill_between (CI). CI shape is partial-check only.
    """
    fmri = make_fmri()
    sns.relplot(data=fmri, x="timepoint", y="signal", kind="line")
    means = fmri.groupby("timepoint")["signal"].mean()
    return [
        {"plt_func": "plot",
         "x": means.index.values.astype(float),
         "y": means.values},
        {"plt_func": "fill_between"},
    ]
