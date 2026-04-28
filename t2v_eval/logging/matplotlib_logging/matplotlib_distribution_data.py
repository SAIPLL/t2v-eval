"""
Monkey-patches for matplotlib distribution / statistical plot functions.

Covers: ``hist``, ``bxp`` (boxplot), ``errorbar``, ``violinplot``, ``violin``,
``eventplot``, ``hist2d``, ``hexbin``, ``pie``.

Call :func:`activate_matplotlib_distribution_data_logging` to install all
patches at once.
"""

from collections import defaultdict

import matplotlib
import matplotlib.axes
import matplotlib.cbook
import matplotlib.container

from t2v_eval.logging.utils import (
    errorbarcontainer_to_logdata,
    eventcollection_to_logdata,
    get_mid_points_of_linecollection,
    get_variable,
    log_data,
    logging_paused,
    patches_to_logdata,
    quadmesh_to_logdata,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_orientation(kwargs: dict) -> str:
    """
    Resolve plot orientation from keyword arguments.

    Handles both the legacy ``vert`` bool and the newer ``orientation`` string.
    Defaults to ``"vertical"``.
    """
    if "vert" in kwargs:
        return "horizontal" if kwargs["vert"] is False else "vertical"
    return kwargs.get("orientation", "vertical")


def _points_to_series(points: list, orientation: str,
                      swap_xy: bool = False,
                      include_orientation: bool = False) -> list:
    """
    Group ``(x, y, color)`` tuples by their position key and build series dicts.

    Points are grouped by x (vertical plots) or y (horizontal plots), then
    sorted within each group before building the series.

    Parameters
    ----------
    points : list of (x, y, color)
    orientation : {"vertical", "horizontal"}
    swap_xy : bool
        When True and orientation is ``"horizontal"``, swap x and y indices.
        Used by violin plots.
    include_orientation : bool
        When True, add ``"orientation"`` to each series dict.
        Used by box plots.
    """
    grouped: dict = defaultdict(list)
    for pt in points:
        key = pt[0] if orientation == "vertical" else pt[1]
        grouped[key].append(pt)

    xi = (1 if (swap_xy and orientation == "horizontal") else 0)
    yi = (0 if (swap_xy and orientation == "horizontal") else 1)

    data_series = []
    for pts in grouped.values():
        pts = sorted(pts)
        series = {
            "x":     [p[xi] for p in pts],
            "y":     [p[yi] for p in pts],
            "color": pts[0][2],
        }
        if include_orientation:
            series["orientation"] = orientation
        data_series.append(series)
    return data_series


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_axes_hist():
    """Patch ``Axes.hist`` to log bar-container data."""
    _orig = matplotlib.axes.Axes.hist

    def hist(self, *args, **kwargs):
        # Force bar histtype so patches is always a BarContainer
        if get_variable("T2V_ISLOG") is True:
            if kwargs.get("histtype", "bar") in ("step", "stepfilled"):
                kwargs["histtype"] = "bar"

        with logging_paused() as was_logging:
            n, bins, patches = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            orientation = kwargs.get("orientation", "vertical")
            data_series  = []
            if isinstance(patches, matplotlib.container.BarContainer):
                data_series += patches_to_logdata(patches, orientation,
                                                  comp_name="hist")
            elif isinstance(patches, list):
                for container in patches:
                    if isinstance(container, matplotlib.container.BarContainer):
                        data_series += patches_to_logdata(container, orientation,
                                                          comp_name="hist")
                    else:
                        raise TypeError(f"Unknown container type: {type(container)}")
            else:
                raise TypeError(f"Unknown patches type: {type(patches)}")
            log_data(self, "hist", args, kwargs, data_series=data_series)

        return n, bins, patches

    matplotlib.axes.Axes.hist = hist


def activate_axes_bxp():
    """Patch ``Axes.bxp`` (the drawing back-end for ``boxplot``) to log box data."""
    _orig = matplotlib.axes.Axes.bxp

    def bxp(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            points = []
            for median in result["medians"]:
                xs, ys = median.get_xdata(), median.get_ydata()
                points.append((float(xs.mean()), float(ys.mean()), median.get_color()))
            for whisker in result["whiskers"]:
                color = whisker.get_color()
                for x, y in zip(whisker.get_xdata(), whisker.get_ydata()):
                    points.append((x, y, color))

            orientation = _get_orientation(kwargs)
            data_series = _points_to_series(points, orientation,
                                            swap_xy=False,
                                            include_orientation=True)
            log_data(self, "bxp", [], {}, data_series)

        return result

    matplotlib.axes.Axes.bxp = bxp


def activate_axes_errorbar():
    """Patch ``Axes.errorbar`` to log error-bar container data."""
    _orig = matplotlib.axes.Axes.errorbar

    def errorbar(self, *args, **kwargs):
        with logging_paused() as was_logging:
            container = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            data_series = errorbarcontainer_to_logdata(container, comp_name="errorbar")
            log_data(self, "errorbar", args, kwargs, data_series=data_series)

        return container

    matplotlib.axes.Axes.errorbar = errorbar


def activate_axes_violinplot():
    """
    Patch ``Axes.violinplot`` to inject quantiles before delegating to ``violin``.

    The actual logging is handled by the ``violin`` patch below.
    """
    _orig = matplotlib.axes.Axes.violinplot

    def violinplot(self, *args, **kwargs):
        if get_variable("T2V_ISLOG") is True:
            X = matplotlib.cbook._reshape_2D(args[0], "X")
            kwargs["quantiles"] = [[0, 0.25, 0.5, 0.75, 1]] * len(X)
        return _orig(self, *args, **kwargs)

    matplotlib.axes.Axes.violinplot = violinplot


def activate_axes_violin():
    """Patch ``Axes.violin`` to log quantile midpoints as box-plot data."""
    _orig = matplotlib.axes.Axes.violin

    def violin(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            points      = get_mid_points_of_linecollection(result.get("cquantiles"))
            orientation = _get_orientation(kwargs)
            data_series = _points_to_series(points, orientation,
                                            swap_xy=True,
                                            include_orientation=False)
            log_data(self, "bxp", [], {}, data_series)

        return result

    matplotlib.axes.Axes.violin = violin


def activate_axes_eventplot():
    """Patch ``Axes.eventplot`` to log events as a single scatter series."""
    _orig = matplotlib.axes.Axes.eventplot

    def eventplot(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            x, y = [], []
            for ec in result:
                for series in eventcollection_to_logdata(ec):
                    x += series["x"]
                    y += series["y"]
            log_data(self, "scatter", [], {},
                     data_series=[{"x": x, "y": y, "label": "event_plot"}])

        return result

    matplotlib.axes.Axes.eventplot = eventplot


def activate_axes_hist2d():
    """Patch ``Axes.hist2d`` to log the resulting ``QuadMesh``."""
    _orig = matplotlib.axes.Axes.hist2d

    def hist2d(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "hist2d", args, kwargs,
                     data_series=quadmesh_to_logdata(result[3]))

        return result

    matplotlib.axes.Axes.hist2d = hist2d


def activate_axes_hexbin():
    """Patch ``Axes.hexbin`` to log hex-bin offsets and counts."""
    _orig = matplotlib.axes.Axes.hexbin

    def hexbin(self, *args, **kwargs):
        with logging_paused() as was_logging:
            collection = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            xy = collection.get_offsets()
            V  = collection.get_array()
            log_data(self, "hexbin", args, kwargs,
                     data_series=[{"V": V, "x": xy[:, 0], "y": xy[:, 1], "z": V}])

        return collection

    matplotlib.axes.Axes.hexbin = hexbin


def activate_axes_pie():
    """Patch ``Axes.pie`` to log slice values and labels."""
    _orig = matplotlib.axes.Axes.pie

    def pie(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            data   = kwargs.get("data")
            x      = args[0]
            labels = kwargs.get("labels")
            if isinstance(x, str):
                x = data.get(x)
                assert x is not None, f"Data for key {x!r} not found in kwargs['data']."
            if isinstance(labels, str):
                labels = data.get(labels)
            log_data(self, "pie", [], kwargs,
                     data_series=[{"x": x, "labels": labels}])

        return result

    matplotlib.axes.Axes.pie = pie

    # ecdf is a thin wrapper around plot — no patch needed.


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_matplotlib_distribution_data_logging():
    """Install all distribution-data monkey-patches on ``matplotlib.axes.Axes``."""
    activate_axes_hist()
    activate_axes_bxp()
    activate_axes_errorbar()
    activate_axes_violinplot()
    activate_axes_violin()
    activate_axes_eventplot()
    activate_axes_hist2d()
    activate_axes_hexbin()
    activate_axes_pie()


