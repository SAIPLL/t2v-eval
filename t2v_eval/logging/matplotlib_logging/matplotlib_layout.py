"""
Monkey-patches for matplotlib layout / axis-annotation functions.

Covers: ``set_title``, ``set_xlabel``, ``set_ylabel``, ``set_xscale``,
``set_yscale``, ``invert_xaxis``, ``invert_yaxis``.

Unlike the data-plot patches, these log *before* calling the original method
(to capture the annotation value), then suppress recursive logging for the
duration of that call.

Call :func:`activate_matplotlib_layout_logging` to install all patches at once.
"""

import matplotlib
import matplotlib.axes

from t2v_eval.logging.utils import get_variable, log_data, set_variable


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_matplotlib_ax_set_title():
    """Patch ``Axes.set_title`` to log the axis title."""
    _orig = matplotlib.axes.Axes.set_title

    def set_title(self, label, *args, **kwargs):
        kwargs["label"] = label
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            log_data(self, "set_title", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.set_title = set_title


def activate_matplotlib_ax_set_xlabel():
    """Patch ``Axes.set_xlabel`` to log the x-axis label."""
    _orig = matplotlib.axes.Axes.set_xlabel

    def set_xlabel(self, xlabel, *args, **kwargs):
        kwargs["xlabel"] = xlabel
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            log_data(self, "set_xlabel", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.set_xlabel = set_xlabel


def activate_matplotlib_ax_set_ylabel():
    """Patch ``Axes.set_ylabel`` to log the y-axis label."""
    _orig = matplotlib.axes.Axes.set_ylabel

    def set_ylabel(self, ylabel, *args, **kwargs):
        kwargs["ylabel"] = ylabel
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            log_data(self, "set_ylabel", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.set_ylabel = set_ylabel


def activate_matplotlib_ax_set_xscale():
    """Patch ``Axes.set_xscale`` to log the x-axis scale (skips ``None``)."""
    _orig = matplotlib.axes.Axes.set_xscale

    def set_xscale(self, value, *args, **kwargs):
        kwargs["value"] = value
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            if value is not None:
                log_data(self, "set_xscale", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.set_xscale = set_xscale


def activate_matplotlib_ax_set_yscale():
    """Patch ``Axes.set_yscale`` to log the y-axis scale (skips ``None``)."""
    _orig = matplotlib.axes.Axes.set_yscale

    def set_yscale(self, value, *args, **kwargs):
        kwargs["value"] = value
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            if value is not None:
                log_data(self, "set_yscale", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.set_yscale = set_yscale


def activate_matplotlib_ax_invert_xaxis():
    """Patch ``Axes.invert_xaxis`` to log axis inversion."""
    _orig = matplotlib.axes.Axes.invert_xaxis

    def invert_xaxis(self, *args, **kwargs):
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            log_data(self, "invert_xaxis", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, *args, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.invert_xaxis = invert_xaxis


def activate_matplotlib_ax_invert_yaxis():
    """Patch ``Axes.invert_yaxis`` to log axis inversion."""
    _orig = matplotlib.axes.Axes.invert_yaxis

    def invert_yaxis(self, *args, **kwargs):
        was_logging = get_variable("T2V_ISLOG")
        if was_logging is True:
            log_data(self, "invert_yaxis", args, kwargs, data_series=[])
            set_variable("T2V_ISLOG", False)
        result = _orig(self, *args, **kwargs)
        set_variable("T2V_ISLOG", was_logging)
        return result

    matplotlib.axes.Axes.invert_yaxis = invert_yaxis


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_matplotlib_layout_logging():
    """Install all layout monkey-patches on ``matplotlib.axes.Axes``."""
    activate_matplotlib_ax_set_title()
    activate_matplotlib_ax_set_xlabel()
    activate_matplotlib_ax_set_ylabel()
    activate_matplotlib_ax_set_xscale()
    activate_matplotlib_ax_set_yscale()
    activate_matplotlib_ax_invert_xaxis()
    activate_matplotlib_ax_invert_yaxis()

    # TODO: set_xlim
    # def set_xlim(self, left=None, right=None, *args, **kwargs):
    #     log_data(self, "set_xlim", left=left, right=right)

    # TODO: set_ylim
    # def set_ylim(self, bottom=None, top=None, *args, **kwargs):
    #     log_data(self, "set_ylim", bottom=bottom, top=top)

    # TODO: set_xticks
    # def set_xticks(self, ticks, labels=None, *args, **kwargs):
    #     log_data(self, "set_xticks", ticks=ticks, labels=labels, **kwargs)

    # TODO: set_yticks
    # def set_yticks(self, ticks, labels=None, *args, **kwargs):
    #     log_data(self, "set_yticks", ticks=ticks, labels=labels, **kwargs)

    # TODO: set_xticklabels
    # def set_xticklabels(self, labels, *args, **kwargs):
    #     log_data(self, "set_xticklabels", labels=labels, **kwargs)

    # TODO: set_yticklabels
    # def set_yticklabels(self, labels, *args, **kwargs):
    #     log_data(self, "set_yticklabels", labels=labels, **kwargs)
