"""
Test cases for ``Axes.contour`` and ``Axes.contourf`` logging.

Strategy: extract from returned ContourSet
-------------------------------------------
``activate_axes_contour / contourf`` delegate to ``contours_to_logdata``,
which iterates ``contours.layers`` and ``contours.allsegs``::

    for layer, segments in zip(contours.layers, contours.allsegs):
        for seg in segments:
            series = {"x": seg[:,0], "y": seg[:,1],
                      "z": np.full(len(seg), layer),
                      "cmin": min(levels), "cmax": max(levels)}

Because the exact contour path coordinates depend on matplotlib's marching-
squares algorithm (not reproducible from numpy alone), expected values are
extracted from the same ``ContourSet`` object that ``ax.contour`` returns.

Series count is variable — one series per disconnected contour segment.
Each series has a constant z (the level value for that segment).
"""

import numpy as np


def _contour_expected(cs, plt_func: str) -> list:
    """
    Compute the expected series list from a ``ContourSet`` object.

    Uses the same ``contours_to_logdata`` formula as the logging patch.

    Parameters
    ----------
    cs : ContourSet
        Object returned by ``ax.contour`` or ``ax.contourf``.
    plt_func : str
        ``"contour"`` or ``"contourf"``.
    """
    cs.axes.figure.canvas.draw()
    levels = cs.levels
    cmin   = float(np.min(levels))
    cmax   = float(np.max(levels))
    expected = []
    for layer, segs in zip(cs.layers, cs.allsegs):
        for seg in segs:
            expected.append({
                "plt_func": plt_func,
                "x": seg[:, 0].tolist(),
                "y": seg[:, 1].tolist(),
                "z": np.full(len(seg), float(layer)).tolist(),
            })
    return expected


# ---------------------------------------------------------------------------
# contour cases
# ---------------------------------------------------------------------------

def case_contour_from_gallery(ax):
    """
    Reproduction of the matplotlib contour gallery example.

    Source (plt.style.use removed):
        X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
        Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
        levels = np.linspace(Z.min(), Z.max(), 7)
        ax.contour(X, Y, Z, levels=levels)
    """
    X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
    levels = np.linspace(Z.min(), Z.max(), 7)

    cs = ax.contour(X, Y, Z, levels=levels)

    return _contour_expected(cs, "contour")


def case_contour_simple(ax):
    """Small grid — fast case verifying the segment structure."""
    X, Y = np.meshgrid(np.linspace(-2, 2, 32), np.linspace(-2, 2, 32))
    Z = X**2 + Y**2   # circular contours

    cs = ax.contour(X, Y, Z, levels=[1.0, 2.0, 3.0])

    return _contour_expected(cs, "contour")


def case_contour_single_level(ax):
    """Single level — simplest possible contour."""
    X, Y = np.meshgrid(np.linspace(-2, 2, 32), np.linspace(-2, 2, 32))
    Z = np.sin(X) * np.cos(Y)

    cs = ax.contour(X, Y, Z, levels=[0.0])

    return _contour_expected(cs, "contour")


# ---------------------------------------------------------------------------
# contourf cases
# ---------------------------------------------------------------------------

def case_contourf_from_gallery(ax):
    """
    Reproduction of the matplotlib contourf gallery example.

    Source (plt.style.use removed):
        X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
        Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
        levels = np.linspace(Z.min(), Z.max(), 7)
        ax.contourf(X, Y, Z, levels=levels)
    """
    X, Y = np.meshgrid(np.linspace(-3, 3, 256), np.linspace(-3, 3, 256))
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
    levels = np.linspace(Z.min(), Z.max(), 7)

    cs = ax.contourf(X, Y, Z, levels=levels)

    return _contour_expected(cs, "contourf")


def case_contourf_simple(ax):
    """Small grid filled contour."""
    X, Y = np.meshgrid(np.linspace(-2, 2, 32), np.linspace(-2, 2, 32))
    Z = X**2 + Y**2

    cs = ax.contourf(X, Y, Z, levels=[0.5, 1.5, 2.5, 3.5])

    return _contour_expected(cs, "contourf")
