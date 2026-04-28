"""
Test cases for ``Axes.imshow`` logging.

Strategy: extract from returned image object
---------------------------------------------
``activate_axes_imshow`` delegates to ``quadmesh_to_logdata``, which reads
``sticky_edges`` and ``get_array()`` from the returned ``AxesImage``.
The same formula is used here to compute expected values::

    V             = img.get_array()               # image data (n_row × n_col)
    x_min, x_max  = sticky_edges.x                # pixel extents
    y_min, y_max  = sticky_edges.y

    col_fracs = linspace(0, 1, n_col)
    row_fracs = linspace(0, 1, n_row)
    jj, ii    = meshgrid(col_fracs, row_fracs)

    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

For a plain ``imshow(Z)`` without an explicit ``extent``, matplotlib sets:
    sticky_edges.x = [-0.5, n_col - 0.5]
    sticky_edges.y = [-0.5, n_row - 0.5]

so the logged x/y span from −0.5 to N − 0.5 (pixel-boundary coordinates,
not pixel centres, due to the linspace interpolation in quadmesh_to_logdata).

Because there is no standalone numpy equivalent of this geometry, expected
values are extracted from the ``AxesImage`` object that ``ax.imshow`` returns
— the same source the logging patch reads.
"""

import numpy as np


def _imshow_expected(img) -> dict:
    """
    Compute the expected ``imshow`` series from the ``AxesImage`` object.

    Uses the same ``quadmesh_to_logdata`` formula as the logging patch.
    """
    img.figure.canvas.draw()          # ensure sticky_edges are finalised
    V = img.get_array()
    x_min, x_max = img.sticky_edges.x[0], img.sticky_edges.x[-1]
    y_min, y_max = img.sticky_edges.y[0], img.sticky_edges.y[-1]
    n_row, n_col = V.shape

    col_fracs = np.linspace(0.0, 1.0, n_col)
    row_fracs = np.linspace(0.0, 1.0, n_row)
    jj, ii    = np.meshgrid(col_fracs, row_fracs)

    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

    return {"plt_func": "imshow",
            "x": xs.tolist(), "y": ys.tolist(), "z": zs.tolist()}


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib imshow gallery example.

    Source (plt.style.use removed):
        X, Y = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))
        Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
        ax.imshow(Z, origin='lower')

    Produces 16×16 = 256 logged (x, y, z) points.
    sticky_edges: x ∈ [-0.5, 15.5],  y ∈ [-0.5, 15.5]
    """
    X, Y = np.meshgrid(np.linspace(-3, 3, 16), np.linspace(-3, 3, 16))
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

    img = ax.imshow(Z, origin='lower')

    return [_imshow_expected(img)]


def case_small_image(ax):
    """3×4 image — verifies grid dimensions and sticky_edges for non-square input."""
    Z = np.array([[1.0, 2.0, 3.0, 4.0],
                  [5.0, 6.0, 7.0, 8.0],
                  [9.0, 10.0, 11.0, 12.0]])

    img = ax.imshow(Z)   # shape (3, 4) → n_row=3, n_col=4

    return [_imshow_expected(img)]


def case_with_explicit_extent(ax):
    """
    imshow with explicit ``extent`` — sticky_edges match the given bounds,
    not the pixel-count defaults.
    """
    Z = np.random.default_rng(0).random((8, 8))

    img = ax.imshow(Z, extent=(-3, 3, -3, 3), origin='lower')

    return [_imshow_expected(img)]
