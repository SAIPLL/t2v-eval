"""
Test cases for ``Axes3D.plot_surface`` logging.

Strategy: extract from returned Poly3DCollection
-------------------------------------------------
``activate_axes3D_plot_surface`` reads mesh vertices from the internal
``_vec`` attribute of the ``Poly3DCollection``::

    x, y, z, _ = poly3d._vec

Because ``_vec`` is computed by matplotlib's surface triangulation and
has no standalone numpy equivalent, expected values are extracted from
the same ``Poly3DCollection`` object that ``ax.plot_surface`` returns.

The logged plt_func is ``"plot_surface3d"``.
"""

import numpy as np


def _surface_expected(poly3d) -> dict:
    """Extract expected x/y/z from the ``Poly3DCollection._vec`` attribute."""
    poly3d.axes.figure.canvas.draw()
    x, y, z, _ = poly3d._vec
    return {"plt_func": "plot_surface3d",
            "x": x, "y": y, "z": z}


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib plot_surface gallery example.

    Source (plt.style.use and imports removed):
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
        ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
    """
    from matplotlib import cm
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    poly3d = ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    return [_surface_expected(poly3d)]


def case_simple_plane(ax):
    """Flat tilted plane — simple surface, easy to reason about."""
    x = np.linspace(0, 4, 8)
    y = np.linspace(0, 4, 8)
    X, Y = np.meshgrid(x, y)
    Z = X + Y

    poly3d = ax.plot_surface(X, Y, Z)

    return [_surface_expected(poly3d)]


def case_paraboloid(ax):
    """Paraboloid surface."""
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    poly3d = ax.plot_surface(X, Y, Z)

    return [_surface_expected(poly3d)]
