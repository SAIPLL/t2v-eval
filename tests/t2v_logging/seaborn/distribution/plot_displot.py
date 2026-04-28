"""
Test cases for ``sns.displot`` (figure-level distribution plots).

Compared to axes-level functions, ``displot``:
  - Creates its own ``FacetGrid`` (``ax`` argument is ignored).
  - Produces **no empty-series artefact** for any configuration.
  - Faceted plots (``col=``) produce one series per facet.

Logged structures
-----------------
    kind="hist"  → "bar" series per hue level (computed via numpy.histogram)
    kind="kde"   → "plot" series per hue level (partial check)
    kind="ecdf"  → "plot" series per hue level (computed)
    x+y hist     → "pcolormesh" (extract from fg.ax)
    x+y kde      → multiple "contour" series (partial check)
"""

import numpy as np
import seaborn as sns

from tests.t2v_logging.helpers import UnorderedSeries, fill_between_expected
from tests.t2v_logging.seaborn.distribution._datasets import (
    make_penguins, make_tips,
)


# ---------------------------------------------------------------------------
# Helpers (same formulas as axes-level)
# ---------------------------------------------------------------------------

def _hist_expected(data, bins) -> dict:
    counts, edges = np.histogram(data, bins=bins)
    return {"plt_func": "bar",
            "x": (edges[:-1] + edges[1:]) / 2,
            "y": counts.astype(float)}


def _ecdf_expected(data) -> dict:
    arr = np.sort(data[~np.isnan(data)])
    n   = len(arr)
    return {"plt_func": "plot",
            "x": np.concatenate([[-np.inf], arr]),
            "y": np.arange(0, n + 1) / n}


# ---------------------------------------------------------------------------
# Histogram cases
# ---------------------------------------------------------------------------

def case_hist_basic(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", bins=20)
    → 1 histogram series.
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", bins=10)
    data = penguins["flipper_length_mm"].dropna().values
    return [_hist_expected(data, bins=10)]


def case_hist_hue(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species")
    → 1 series per species.  Global bin edges, UnorderedSeries for matching.
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", hue="species", bins=10)
    _, global_edges = np.histogram(
        penguins["flipper_length_mm"].dropna(), bins=10)
    centers = (global_edges[:-1] + global_edges[1:]) / 2
    expected = UnorderedSeries()
    for sp in penguins["species"].unique():
        data = penguins.loc[penguins["species"] == sp,
                            "flipper_length_mm"].dropna().values
        counts, _ = np.histogram(data, bins=global_edges)
        expected.append({"plt_func": "bar",
                         "x": centers, "y": counts.astype(float)})
    return expected


def case_hist_facet_col(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", col="sex")
    → 1 series per sex.  Global bin edges shared across all facets.
    UnorderedSeries since facet order is seaborn-controlled.
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", col="sex", bins=10)
    # Seaborn computes bin edges from the full data, applies them per facet
    _, global_edges = np.histogram(
        penguins["flipper_length_mm"].dropna(), bins=10)
    centers = (global_edges[:-1] + global_edges[1:]) / 2
    expected = UnorderedSeries()
    for sx in penguins["sex"].unique():
        data = penguins.loc[penguins["sex"] == sx,
                            "flipper_length_mm"].dropna().values
        counts, _ = np.histogram(data, bins=global_edges)
        expected.append({"plt_func": "bar",
                         "x": centers, "y": counts.astype(float)})
    return expected


# ---------------------------------------------------------------------------
# KDE cases
# ---------------------------------------------------------------------------

def case_kde_basic(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", kind="kde")
    → 1 KDE line series (partial check).
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", kind="kde")
    return [{"plt_func": "plot"}]


def case_kde_hue(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")
    → 1 KDE line per species (partial check).
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")
    n_hue = penguins["species"].nunique()
    return [{"plt_func": "plot"}] * n_hue


def case_kde_fill_hue(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)
    → 1 fill_between per species.
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", hue="species",
                kind="kde", fill=True)
    n_hue = penguins["species"].nunique()
    return [{"plt_func": "fill_between"}] * n_hue


# ---------------------------------------------------------------------------
# ECDF cases
# ---------------------------------------------------------------------------

def case_ecdf_basic(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", kind="ecdf")
    → 1 ECDF line (values checked).
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", kind="ecdf")
    return [_ecdf_expected(penguins["flipper_length_mm"].values)]


def case_ecdf_hue(ax):
    """
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf")
    → 1 ECDF per species.  UnorderedSeries with full values.
    """
    penguins = make_penguins()
    sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf")
    expected = UnorderedSeries()
    for sp in penguins["species"].unique():
        data = penguins.loc[penguins["species"] == sp,
                            "flipper_length_mm"].values
        expected.append(_ecdf_expected(data))
    return expected


# ---------------------------------------------------------------------------
# 2-D distribution cases
# ---------------------------------------------------------------------------

def case_2d_hist(ax):
    """
    Tutorial: sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm")
    → 1 "pcolormesh" series (partial check — seaborn's colormap kwarg
    interferes with the pcolormesh logging pipeline).
    """
    penguins = make_penguins()
    sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", bins=5)
    return [{"plt_func": "pcolormesh"}]


def case_2d_kde(ax):
    """
    Tutorial: sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde")
    → Multiple "contour" series (partial check — count + plt_func).
    """
    penguins = make_penguins()
    sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde")
    # Number of contour lines depends on data; just verify all are "contour"
    # Return a sentinel: test runner will check after we count logged series
    return None   # golden-path: inspect n_series at runtime


def case_2d_kde_hue(ax):
    """
    Tutorial: sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm",
                          hue="species", kind="kde")
    → Multiple "contour" series (partial check).
    """
    penguins = make_penguins()
    sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm",
                hue="species", kind="kde")
    return None  # golden-path
