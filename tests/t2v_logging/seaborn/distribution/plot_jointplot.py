"""
Test cases for ``sns.jointplot`` logging.

How it is logged
----------------
``jointplot`` creates a figure with three axes (joint + two marginals).
Each axes produces its own logged series:

    kind="scatter":
        [barh (y marginal), bar (x marginal), scatter (joint)]

    kind="kde":
        [multiple contour/fill_between from joint KDE,
         KDE lines from marginals]

Strategy
--------
- **scatter**: joint scatter → computed (x/y values match input columns).
  Marginal histograms → partial check (plt_func only).
- **kde**: all partial checks (contour geometry is complex).
"""

import seaborn as sns

from tests.t2v_logging.seaborn.distribution._datasets import make_penguins


def case_scatter(ax):
    """
    Tutorial: sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")

    Produces 3 series: scatter (joint) + bar (x marginal) + barh (y marginal).
    The logging order depends on matplotlib axes object IDs (non-deterministic
    across runs), so golden-path is used: verifies 3 series, all with valid
    distribution plt_funcs, all length-consistent.
    """
    penguins = make_penguins()
    sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")
    return None   # golden-path


def case_kde(ax):
    """
    Tutorial: sns.jointplot(data=penguins, x="bill_length_mm",
                            y="bill_depth_mm", kind="kde")

    Joint KDE → many contour/fill_between; marginals → KDE lines.
    All partial checks.
    """
    penguins = make_penguins()
    sns.jointplot(data=penguins, x="bill_length_mm",
                  y="bill_depth_mm", kind="kde")
    return None   # golden-path: all partial
