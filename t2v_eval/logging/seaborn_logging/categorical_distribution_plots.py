"""
Monkey-patches for seaborn categorical distribution plots.

Covers ``_CategoricalPlotter.plot_boxens`` (boxen plot),
``_CategoricalPlotter.plot_boxes`` (box plot),
``_CategoricalPlotter.plot_violins`` (violin plot), and
``_CategoricalPlotter.plot_errorbars``.

The first three share the same post-call data-extraction helper
:func:`_extract_cat_dist_data` and log under the ``"bxp"`` plt_func.
The errorbars patch suppresses cap-size during logging to avoid
double-counting those artists.

Call :func:`activate_sns_categorical_distribution_plots` to install all patches.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn.categorical

from t2v_eval.logging.utils import get_variable, log_data, logging_paused


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_cat_dist_data(plotter, kwargs: dict) -> dict:
    """
    Extract per-axes distribution data from a seaborn categorical plotter.

    For each sub-group, computes the five-number summary (0, 25, 50, 75, 100th
    percentiles) of the distribution variable and records position, orientation,
    colour, and axis-inversion flags.

    Parameters
    ----------
    plotter : seaborn.categorical._CategoricalPlotter
    kwargs : dict
        Keyword arguments originally passed to the ``plot_*`` method
        (used to read ``width``, ``dodge``, ``gap``).

    Returns
    -------
    dict
        ``{ax: [series_dict, …]}`` — one entry per axes object.
    """
    width  = kwargs.get("width", 0.8)
    dodge  = kwargs.get("dodge", True)
    gap    = kwargs.get("gap", 0.0)

    iter_vars = [plotter.orient, "hue"]
    axes_data: dict = defaultdict(list)

    for sub_vars, sub_data in plotter.iter_data(iter_vars,
                                                from_comp_data=True,
                                                allow_empty=False):
        ax = plotter._get_axes(sub_vars)

        # Resolve the position centre with optional dodge / gap
        pos_data = pd.DataFrame({
            plotter.orient: [sub_vars[plotter.orient]],
            "width":        [width * plotter._native_width],
        })
        if dodge:
            plotter._dodge(sub_vars, pos_data)
        if gap:
            pos_data["width"] *= 1 - gap
        pos_center = round(pos_data[plotter.orient].item(), 1)

        # Identify the distribution column (everything that isn't a grouping var)
        dist_key = list(set(sub_data.columns) - set(sub_vars.keys()))
        assert len(dist_key) == 1, (
            f"Expected exactly one distribution column, got {len(dist_key)}.\n"
            f"sub_data columns: {list(sub_data.columns)}\n"
            f"sub_vars keys:    {list(sub_vars.keys())}"
        )
        dist_col  = dist_key[0]
        quantiles = np.quantile(sub_data[dist_col], [0, 0.25, 0.5, 0.75, 1])

        color = plotter._hue_map(sub_vars["hue"]) if "hue" in sub_vars else None
        series = sub_vars.copy()
        series.update({
            dist_col:            quantiles,
            plotter.orient:      [pos_center] * len(quantiles),
            "orientation":       "horizontal" if plotter.orient == "y" else "vertical",
            "color":             color,
            "x_axis_inverted":   bool(ax.get_xaxis().get_inverted()),
            "y_axis_inverted":   bool(ax.get_yaxis().get_inverted()),
        })
        axes_data[ax].append(series)

    return axes_data


def _log_cat_dist(plotter, kwargs: dict) -> None:
    """
    Extract distribution data and write one ``"bxp"`` log entry per axes.

    Parameters
    ----------
    plotter : seaborn.categorical._CategoricalPlotter
    kwargs : dict
        Keyword arguments from the calling patch function.
    """
    for ax, data_series in _extract_cat_dist_data(plotter, kwargs).items():
        ax.figure.canvas.draw()
        log_data(ax, "bxp", [], {}, data_series=data_series)


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_sns_boxenplot() -> None:
    """Patch ``_CategoricalPlotter.plot_boxens`` to log boxen-plot data."""
    _orig = seaborn.categorical._CategoricalPlotter.plot_boxens

    def plot_boxens(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)
        if was_logging:
            _log_cat_dist(self, kwargs)
        return result

    seaborn.categorical._CategoricalPlotter.plot_boxens = plot_boxens


def activate_sns_boxplot() -> None:
    """Patch ``_CategoricalPlotter.plot_boxes`` to log box-plot data."""
    _orig = seaborn.categorical._CategoricalPlotter.plot_boxes

    def plot_boxes(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)
        if was_logging:
            _log_cat_dist(self, kwargs)
        return result

    seaborn.categorical._CategoricalPlotter.plot_boxes = plot_boxes


def activate_sns_violinplot() -> None:
    """Patch ``_CategoricalPlotter.plot_violins`` to log violin-plot data."""
    _orig = seaborn.categorical._CategoricalPlotter.plot_violins

    def plot_violins(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)
        if was_logging:
            _log_cat_dist(self, kwargs)
        return result

    seaborn.categorical._CategoricalPlotter.plot_violins = plot_violins


def activate_sns_plot_errorbars() -> None:
    """Patch ``_CategoricalPlotter.plot_errorbars`` to suppress cap-size during logging."""
    _orig = seaborn.categorical._CategoricalPlotter.plot_errorbars

    def plot_errorbars(self, ax, data, capsize, err_kws):
        if get_variable("T2V_ISLOG") is True:
            capsize = 0
        return _orig(self, ax, data, capsize, err_kws)

    seaborn.categorical._CategoricalPlotter.plot_errorbars = plot_errorbars


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_sns_categorical_distribution_plots() -> None:
    """Install all categorical-distribution monkey-patches on ``seaborn.categorical``."""
    activate_sns_boxenplot()
    activate_sns_boxplot()
    activate_sns_violinplot()
    activate_sns_plot_errorbars()
