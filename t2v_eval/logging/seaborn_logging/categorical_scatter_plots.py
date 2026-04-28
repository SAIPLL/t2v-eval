"""
Monkey-patches for seaborn categorical scatter plots.

Covers ``_CategoricalPlotter.plot_strips`` (strip plot) and
``_CategoricalPlotter.plot_swarms`` (swarm plot).

Both patches share the same post-call data-extraction helper
:func:`_extract_cat_scatter_data`.

Call :func:`activate_sns_categorical_scatter_plots` to install all patches.
"""

from collections import defaultdict

import seaborn.categorical

from t2v_eval.logging.utils import log_data, logging_paused


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_cat_scatter_data(plotter, kwargs: dict) -> dict:
    """
    Extract per-axes scatter data from a seaborn categorical plotter.

    Iterates over the plotter's sub-data groups, applies dodge offsets and
    inverse scaling, and collects ``(x, y, s, color)`` series grouped by axes.

    Parameters
    ----------
    plotter : seaborn.categorical._CategoricalPlotter
        The plotter instance whose ``comp_data`` and mappings are used.
    kwargs : dict
        The keyword arguments originally passed to ``plot_strips`` /
        ``plot_swarms`` (used to read ``dodge`` and ``plot_kws``).

    Returns
    -------
    dict
        ``{ax: [series_dict, …]}`` — one entry per axes object.
    """
    dodge   = kwargs.get("dodge")
    width   = 0.8 * plotter._native_width
    offsets = plotter._nested_offsets(width, dodge)

    iter_vars = [plotter.orient]
    if dodge:
        iter_vars.append("hue")

    axes_data: dict = defaultdict(list)

    for sub_vars, sub_data in plotter.iter_data(iter_vars,
                                                from_comp_data=True,
                                                allow_empty=False):
        ax = plotter._get_axes(sub_vars)

        if offsets is not None:
            dodge_move = offsets[sub_data["hue"].map(plotter._hue_map.levels.index)]
            if not sub_data.empty:
                sub_data[plotter.orient] = sub_data[plotter.orient] + dodge_move

        plotter._invert_scale(ax, sub_data)

        s = kwargs.get("plot_kws", {}).get("s", 5)
        color = plotter._hue_map(sub_data["hue"]) if "hue" in plotter.variables else None
        axes_data[ax].append({
            "x":               sub_data["x"],
            "y":               sub_data["y"],
            "s":               [s] * len(sub_data["x"]),
            "color":           color,
            "x_axis_inverted": bool(ax.get_xaxis().get_inverted()),
            "y_axis_inverted": bool(ax.get_yaxis().get_inverted()),
        })

    return axes_data


def _log_cat_scatter(plotter, kwargs: dict) -> None:
    """
    Extract scatter data and write one log entry per axes.

    Parameters
    ----------
    plotter : seaborn.categorical._CategoricalPlotter
    kwargs : dict
        Keyword arguments from the calling patch function.
    """
    for ax, data_series in _extract_cat_scatter_data(plotter, kwargs).items():
        ax.figure.canvas.draw()
        log_data(ax, "scatter", [], {}, data_series=data_series)


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_sns_stripplot() -> None:
    """Patch ``_CategoricalPlotter.plot_strips`` to log strip-plot data."""
    _orig = seaborn.categorical._CategoricalPlotter.plot_strips

    def plot_strips(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)
        if was_logging:
            _log_cat_scatter(self, kwargs)
        return result

    seaborn.categorical._CategoricalPlotter.plot_strips = plot_strips


def activate_sns_swarmplot() -> None:
    """Patch ``_CategoricalPlotter.plot_swarms`` to log swarm-plot data."""
    _orig = seaborn.categorical._CategoricalPlotter.plot_swarms

    def plot_swarms(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)
        if was_logging:
            _log_cat_scatter(self, kwargs)
        return result

    seaborn.categorical._CategoricalPlotter.plot_swarms = plot_swarms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_sns_categorical_scatter_plots() -> None:
    """Install all categorical-scatter monkey-patches on ``seaborn.categorical``."""
    activate_sns_stripplot()
    activate_sns_swarmplot()
