"""
Monkey-patches for matplotlib 2-D ("pairwise") plot functions.

Each ``activate_axes_*`` function wraps one ``matplotlib.axes.Axes`` method so
that, when T2V logging is active, the plotted data are serialised to disk via
:func:`~t2v_eval.logging.utils.log_data`.

Call :func:`activate_matplotlib_pairwise_data_logging` to install all patches
at once.
"""

import matplotlib
import matplotlib.axes
import matplotlib.dates as mdates
import numpy as np

from t2v_eval.logging.utils import (
    errorbarcontainer_to_logdata,
    get_variable,
    line2d_to_logdata,
    linecollection_to_logdata,
    log_data,
    logging_paused,
    patches_to_logdata,
    polycollection_to_logdata,
)


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_axes_plot():
    """Patch ``Axes.plot`` to log Line2D data."""
    _orig = matplotlib.axes.Axes.plot

    def plot(self, *args, **kwargs):
        lines = _orig(self, *args, **kwargs)
        if get_variable("T2V_ISLOG") is True:
            self.figure.canvas.draw()
            try:
                is_x_dt = isinstance(self.xaxis.get_major_locator(), mdates.AutoDateLocator)
                is_y_dt = isinstance(self.yaxis.get_major_locator(), mdates.AutoDateLocator)
            except Exception:
                is_x_dt = is_y_dt = False
            data_series = []
            for line in lines:
                data_series += line2d_to_logdata(line, is_x_datetime=is_x_dt,
                                                       is_y_datetime=is_y_dt)
            log_data(self, "plot", [], {}, data_series)
        return lines

    matplotlib.axes.Axes.plot = plot


def activate_axes_scatter():
    """Patch ``Axes.scatter`` to log PathCollection data."""
    _orig = matplotlib.axes.Axes.scatter

    def scatter(self, *args, **kwargs):
        with logging_paused() as was_logging:
            collection = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            x_data = collection.get_offsets()[:, 0]
            y_data = collection.get_offsets()[:, 1]
            size   = collection.get_sizes()
            color  = collection.get_facecolor()
            n      = len(x_data)
            if   len(size)  == 1: size  = [size[0]]  * n
            elif len(size)  != n: size  = None          # length mismatch — drop
            if   len(color) == 1: color = [color[0]] * n
            elif len(color) != n: color = None          # length mismatch — drop
            data_series = [{
                "x":      x_data,
                "y":      y_data,
                "marker": kwargs.get("marker"),
                "s":      size,
                "color":  color,
                "name":   kwargs.get("label"),
            }]
            log_data(self, "scatter", [], {}, data_series=data_series)
        return collection

    matplotlib.axes.Axes.scatter = scatter


def activate_axes_bar():
    """Patch ``Axes.bar`` to log vertical bar (and optional errorbar) data."""
    _orig = matplotlib.axes.Axes.bar

    def bar(self, *args, **kwargs):
        with logging_paused() as was_logging:
            bar_container = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "bar", args, kwargs,
                     data_series=patches_to_logdata(bar_container.patches,
                                                    orientation="vertical",
                                                    comp_name="bar"))
            if bar_container.errorbar:
                log_data(self, "errorbar", args, kwargs,
                         data_series=errorbarcontainer_to_logdata(bar_container.errorbar,
                                                                  comp_name="bar"))
        return bar_container

    matplotlib.axes.Axes.bar = bar


def activate_axes_barh():
    """Patch ``Axes.barh`` to log horizontal bar (and optional errorbar) data."""
    _orig = matplotlib.axes.Axes.barh

    def barh(self, *args, **kwargs):
        with logging_paused() as was_logging:
            bar_container = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "barh", args, kwargs,
                     data_series=patches_to_logdata(bar_container.patches,
                                                    orientation="horizontal",
                                                    comp_name="barh"))
            if bar_container.errorbar:
                log_data(self, "errorbar", args, kwargs,
                         data_series=errorbarcontainer_to_logdata(bar_container.errorbar,
                                                                  comp_name="barh"))
        return bar_container

    matplotlib.axes.Axes.barh = barh


def activate_axes_stem():
    """Patch ``Axes.stem`` to log stem-plot line data."""
    _orig = matplotlib.axes.Axes.stem

    def stem(self, *args, **kwargs):
        with logging_paused() as was_logging:
            stem_container = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            data_series = (
                line2d_to_logdata(stem_container.markerline)
                + linecollection_to_logdata(stem_container.stemlines)
                + line2d_to_logdata(stem_container.baseline)
            )
            log_data(self, "stem", args, kwargs, data_series=data_series)
        return stem_container

    matplotlib.axes.Axes.stem = stem


def activate_axes_fill_between():
    """Patch ``Axes.fill_between`` to log polygon vertex data."""
    _orig = matplotlib.axes.Axes.fill_between

    def fill_between(self, *args, **kwargs):
        with logging_paused() as was_logging:
            poly = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "fill_between", [], {},
                     data_series=polycollection_to_logdata(poly))
        return poly

    matplotlib.axes.Axes.fill_between = fill_between


def activate_axes_fill_betweenx():
    """Patch ``Axes.fill_betweenx`` to log polygon vertex data."""
    _orig = matplotlib.axes.Axes.fill_betweenx

    def fill_betweenx(self, *args, **kwargs):
        with logging_paused() as was_logging:
            poly = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "fill_betweenx", [], {},
                     data_series=polycollection_to_logdata(poly))
        return poly

    matplotlib.axes.Axes.fill_betweenx = fill_betweenx

    # TODO: stackplot (~fill_between × n)


def activate_axes_stairs():
    """Patch ``Axes.stairs`` to log stair-step (x, y) data.

    Each step is represented by its centre x position paired with its height.
    For ``orientation="vertical"`` (default): ``x`` = step centres, ``y`` = values.
    For ``orientation="horizontal"``:          ``x`` = values,        ``y`` = step centres.
    """
    _orig = matplotlib.axes.Axes.stairs

    def stairs(self, values, edges=None, *,
               orientation="vertical", baseline=0, fill=False, **kwargs):
        kw = dict(values=values, orientation=orientation, baseline=baseline,
                  fill=fill, **kwargs)
        if edges is not None:
            kw["edges"] = edges

        with logging_paused() as was_logging:
            result = _orig(self, **kw)

        if was_logging:
            self.figure.canvas.draw()
            data    = result.get_data()
            vals    = np.array(data.values, dtype=float)
            edges_  = np.array(data.edges,  dtype=float)
            centers = (edges_[:-1] + edges_[1:]) / 2
            if orientation == "vertical":
                series = {"x": centers, "y": vals}
            else:
                series = {"x": vals, "y": centers}
            log_data(self, "stairs", [], kw, data_series=[series])

        return result

    matplotlib.axes.Axes.stairs = stairs


def activate_axes_axhline():
    """Patch ``Axes.axhline`` to log horizontal reference line data."""
    _orig = matplotlib.axes.Axes.axhline

    def axhline(self, *args, **kwargs):
        with logging_paused() as was_logging:
            line2d = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "axhline", [], {}, data_series=line2d_to_logdata(line2d))
        return line2d

    matplotlib.axes.Axes.axhline = axhline


def activate_axes_axvline():
    """Patch ``Axes.axvline`` to log vertical reference line data."""
    _orig = matplotlib.axes.Axes.axvline

    def axvline(self, *args, **kwargs):
        with logging_paused() as was_logging:
            line2d = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "axvline", [], {}, data_series=line2d_to_logdata(line2d))
        return line2d

    matplotlib.axes.Axes.axvline = axvline

    # TODO: vlines, hlines, axvspan, axhspan


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_matplotlib_pairwise_data_logging():
    """Install all pairwise-data monkey-patches on ``matplotlib.axes.Axes``."""
    activate_axes_plot()
    activate_axes_scatter()
    activate_axes_bar()
    activate_axes_barh()
    activate_axes_stem()
    activate_axes_fill_between()
    activate_axes_fill_betweenx()
    activate_axes_stairs()
    activate_axes_axhline()
    activate_axes_axvline()
