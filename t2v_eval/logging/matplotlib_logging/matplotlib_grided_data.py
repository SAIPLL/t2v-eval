"""
Monkey-patches for matplotlib gridded / matrix plot functions.

Covers: ``imshow``, ``pcolormesh``, ``Figure.colorbar``, ``contour``,
``contourf``.

Call :func:`activate_matplotlib_gridded_data_logging` to install all patches
at once.
"""

import matplotlib
import matplotlib.axes
import matplotlib.figure
from mpl_toolkits.mplot3d import Axes3D

from t2v_eval.logging.utils import (
    contours_to_logdata,
    log_data,
    logging_paused,
    quadmesh_to_logdata,
)


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_axes_imshow():
    """Patch ``Axes.imshow`` to log the resulting ``QuadMesh``."""
    _orig = matplotlib.axes.Axes.imshow

    def imshow(self, *args, **kwargs):
        with logging_paused() as was_logging:
            img = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "imshow", [], kwargs,
                     data_series=quadmesh_to_logdata(img))
        return img

    matplotlib.axes.Axes.imshow = imshow


def activate_axes_pcolormesh():
    """Patch ``Axes.pcolormesh`` to log the resulting ``QuadMesh``."""
    _orig = matplotlib.axes.Axes.pcolormesh

    def pcolormesh(self, *args, **kwargs):
        with logging_paused() as was_logging:
            result = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "pcolormesh", [], kwargs,
                     data_series=quadmesh_to_logdata(result))
        return result

    matplotlib.axes.Axes.pcolormesh = pcolormesh


def activate_figure_colorbar():
    """Patch ``Figure.colorbar`` to suppress recursive logging."""
    _orig = matplotlib.figure.Figure.colorbar

    def colorbar(self, *args, **kwargs):
        with logging_paused():
            result = _orig(self, *args, **kwargs)
        return result

    matplotlib.figure.Figure.colorbar = colorbar


def activate_axes_contour():
    """Patch ``Axes.contour`` to log contour-set data (2-D axes only)."""
    _orig = matplotlib.axes.Axes.contour

    def contour(self, *args, **kwargs):
        if isinstance(self, Axes3D):
            return _orig(self, *args, **kwargs)
        with logging_paused() as was_logging:
            contours = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "contour", args, kwargs,
                     data_series=contours_to_logdata(contours))
        return contours

    matplotlib.axes.Axes.contour = contour


def activate_axes_contourf():
    """Patch ``Axes.contourf`` to log filled contour-set data (2-D axes only)."""
    _orig = matplotlib.axes.Axes.contourf

    def contourf(self, *args, **kwargs):
        if isinstance(self, Axes3D):
            return _orig(self, *args, **kwargs)
        with logging_paused() as was_logging:
            contours = _orig(self, *args, **kwargs)
        if was_logging:
            self.figure.canvas.draw()
            log_data(self, "contourf", args, kwargs,
                     data_series=contours_to_logdata(contours))
        return contours

    matplotlib.axes.Axes.contourf = contourf


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_matplotlib_gridded_data_logging():
    """Install all gridded-data monkey-patches on ``matplotlib.axes.Axes``."""
    activate_figure_colorbar()
    activate_axes_imshow()
    activate_axes_pcolormesh()
    activate_axes_contour()
    activate_axes_contourf()
