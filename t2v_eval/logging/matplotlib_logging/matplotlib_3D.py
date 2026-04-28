"""
Monkey-patches for mpl_toolkits 3-D plot functions.

Covers: ``Axes3D.scatter`` / ``scatter3D``, ``Axes3D.bar3d``,
``Axes3D.plot_surface``.

Call :func:`activate_matplotlib_3D_data_logging` to install all patches
at once.
"""

import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D

from t2v_eval.logging.utils import log_data, logging_paused


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_axes3D_scatter():
    """Patch ``Axes3D.scatter`` (and ``scatter3D``) to log 3-D point-cloud data."""
    _orig = Axes3D.scatter

    def scatter(self, *args, **kwargs):
        with logging_paused() as was_logging:
            collection = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            x_data, y_data, z_data = collection._offsets3d
            size  = collection.get_sizes()
            color = collection.get_facecolor()
            n     = len(x_data)
            if   len(size)  == 1: size  = [size[0]]  * n
            elif len(size)  != n: size  = None          # length mismatch — drop
            if   len(color) == 1: color = [color[0]] * n
            elif len(color) != n: color = None          # length mismatch — drop
            data_series = [{
                "x":      x_data,
                "y":      y_data,
                "z":      z_data,
                "marker": kwargs.get("marker"),
                "s":      size,
                "color":  color,
                "name":   kwargs.get("label"),
            }]
            log_data(self, "scatter3d", [], {}, data_series=data_series)

        return collection

    Axes3D.scatter  = scatter
    Axes3D.scatter3D = scatter


def activate_axes3D_bar3d():
    """Patch ``Axes3D.bar3d`` to log 3-D bar top-corner positions."""
    _orig = Axes3D.bar3d

    def bar3d(self, x, y, z, dx, dy, dz, **kwargs):
        with logging_paused() as was_logging:
            collection = _orig(self, x, y, z, dx, dy, dz, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            data_series = [{
                "x":     np.array(x) + np.array(dx),
                "y":     np.array(y) + np.array(dy),
                "z":     np.array(z) + np.array(dz),
                "color": kwargs.get("color"),
                "name":  kwargs.get("label"),
            }]
            log_data(self, "bar3d", [x, y, z, dx, dy, dz], kwargs,
                     data_series=data_series)

        return collection

    Axes3D.bar3d = bar3d


def activate_axes3D_plot_surface():
    """Patch ``Axes3D.plot_surface`` to log surface mesh vertices."""
    _orig = Axes3D.plot_surface

    def plot_surface(self, *args, **kwargs):
        with logging_paused() as was_logging:
            poly3d = _orig(self, *args, **kwargs)

        if was_logging:
            self.figure.canvas.draw()
            x, y, z, _ = poly3d._vec
            data_series = [{
                "x":    x,
                "y":    y,
                "z":    z,
                "name": kwargs.get("label"),
            }]
            log_data(self, "plot_surface3d", args, kwargs, data_series=data_series)

        return poly3d

    Axes3D.plot_surface = plot_surface


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_matplotlib_3D_data_logging():
    """Install all 3-D data monkey-patches on ``mpl_toolkits.mplot3d.axes3d.Axes3D``."""
    activate_axes3D_scatter()
    activate_axes3D_bar3d()
    activate_axes3D_plot_surface()

    # TODO: fill_between3d
    # TODO: fill_betweenx3d
    # TODO: plot3d
    # TODO: quiver3d
    # TODO: stem3d
    # TODO: plot_trisurf
    # TODO: voxels
    # TODO: plot_wireframe
