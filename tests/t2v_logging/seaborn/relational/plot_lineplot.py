"""
Test cases for ``sns.lineplot`` / ``sns.relplot(kind="line")`` logging.

How it is logged
----------------
``sns.lineplot`` draws using matplotlib's ``ax.plot()``, which IS patched.
When aggregation is enabled, the CI band is drawn with ``ax.fill_between()``,
which is also patched.

Series structure by configuration
-----------------------------------
    estimator=None, no hue → [empty "plot", all-data "plot"]
    errorbar=None,  no hue → [empty "plot", mean "plot"]
    default CI,     no hue → [empty "plot", mean "plot", "fill_between"]
    errorbar=None, hue=X   → [line_A "plot", line_B "plot"]
    default CI,    hue=X   → [line_A "plot", fill_A, line_B "plot", fill_B]

The empty "plot" series (n=0) appears when there is **no** hue semantic,
mirroring the scatter behaviour.

Expected values
---------------
- ``estimator=None``: all raw data logged — partial check only (plt_func + length).
- ``errorbar=None``: mean per timepoint — computed via ``groupby.mean()``.
- CI band (fill_between): complex shape — partial check only (plt_func + length).
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import fill_between_expected
from tests.t2v_logging.seaborn.relational._datasets import make_fmri


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_series(df, x_col: str, y_col: str,
                 plt_func: str = "plot") -> dict:
    """Compute expected mean line: x = sorted unique timepoints, y = means."""
    means = df.groupby(x_col)[y_col].mean()
    return {
        "plt_func": plt_func,
        "x": means.index.values.astype(float),
        "y": means.values,
    }


def _partial(plt_func: str) -> dict:
    """Partial check — only plt_func + length consistency, no value check."""
    return {"plt_func": plt_func}


# ---------------------------------------------------------------------------
# Test cases — no hue
# ---------------------------------------------------------------------------

def case_estimator_none_no_hue(ax):
    """
    All raw data plotted with no aggregation, no hue.
    Tutorial: sns.relplot(..., kind="line", estimator=None)

    → 2 series: empty "plot" + all-data "plot" (n = total rows).
    Value check skipped for the data series (seaborn's internal ordering).
    """
    fmri = make_fmri()
    sns.lineplot(data=fmri, x="timepoint", y="signal",
                 estimator=None, ax=ax)
    return [
        _partial("plot"),                    # empty legend artefact
        _partial("plot"),                    # all raw data points
    ]


def case_errorbar_none_no_hue(ax):
    """
    Mean line only, no CI, no hue.
    Tutorial: sns.relplot(..., kind="line", errorbar=None)

    → 2 series: empty "plot" + mean line (n = 18 timepoints).
    """
    fmri = make_fmri()
    sns.lineplot(data=fmri, x="timepoint", y="signal",
                 errorbar=None, ax=ax)
    return [
        _partial("plot"),                    # empty legend artefact
        _mean_series(fmri, "timepoint", "signal"),   # mean line, values checked
    ]


def case_with_ci_no_hue(ax):
    """
    Mean line + 95% CI band, no hue.
    Tutorial: sns.relplot(..., kind="line")  ← default

    → 3 series: empty "plot" + mean "plot" + "fill_between" CI band.
    CI band shape is complex — partial check only.
    """
    fmri = make_fmri()
    sns.lineplot(data=fmri, x="timepoint", y="signal", ax=ax)
    return [
        _partial("plot"),                            # empty artefact
        _mean_series(fmri, "timepoint", "signal"),  # mean line
        _partial("fill_between"),                    # CI band (shape varies)
    ]


# ---------------------------------------------------------------------------
# Test cases — with hue
# ---------------------------------------------------------------------------

def case_errorbar_none_with_hue(ax):
    """
    One mean line per hue level, no CI.
    Tutorial: sns.relplot(..., kind="line", hue="event", errorbar=None)

    → 2 series: one mean line per event ("stim", "cue").
    No empty series when hue is present.
    """
    fmri = make_fmri()
    sns.lineplot(data=fmri, x="timepoint", y="signal",
                 hue="event", errorbar=None, ax=ax)

    def _event_mean(event):
        return _mean_series(
            fmri[fmri["event"] == event], "timepoint", "signal")

    # Series order matches the order of first appearance in the data
    events = list(fmri["event"].unique())
    return [_event_mean(ev) for ev in events]


def case_with_ci_and_hue(ax):
    """
    Mean line + CI per hue level.
    Tutorial: sns.relplot(..., kind="line", hue="event")  ← default

    → 4 series: (line + fill_between) × 2 events.
    """
    fmri = make_fmri()
    sns.lineplot(data=fmri, x="timepoint", y="signal",
                 hue="event", ax=ax)

    events = list(fmri["event"].unique())   # order of first appearance
    expected = []
    for ev in events:
        expected.append(
            _mean_series(fmri[fmri["event"] == ev], "timepoint", "signal")
        )
        expected.append(_partial("fill_between"))   # CI band
    return expected
