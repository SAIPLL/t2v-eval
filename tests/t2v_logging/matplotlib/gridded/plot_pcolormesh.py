"""
Test cases for ``Axes.pcolormesh`` logging.

Strategy: extract from returned QuadMesh object
-------------------------------------------------
``activate_axes_pcolormesh`` delegates to ``quadmesh_to_logdata``, which reads
``sticky_edges`` and ``get_array()`` from the returned ``QuadMesh``.
The same formula is used here::

    V             = qm.get_array()                # cell values
    x_min, x_max  = sticky_edges.x
    y_min, y_max  = sticky_edges.y

    col_fracs = linspace(0, 1, n_col)
    row_fracs = linspace(0, 1, n_row)
    jj, ii    = meshgrid(col_fracs, row_fracs)

    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

Sticky-edges for vertex-based X, Y
------------------------------------
When X and Y are 2-D vertex coordinate arrays (not cell centres), matplotlib
centres each cell on its vertex and extends the mesh by half a bin-width on
each side.  For uneven sampling, this produces non-symmetric sticky edges
that extend *beyond* the input coordinate range:

    example x = [-3, -2, ..., 2.3, 3]  (16 vertices, uneven spacing)
    sticky x   = [-3.5, 3.35]          (extended by half the boundary step)

Because the exact sticky-edge values depend on matplotlib's internal
mesh-edge computation, expected values are extracted from the ``QuadMesh``
object that ``ax.pcolormesh`` returns — the same source the logging patch
reads.
"""

import numpy as np


def _pcolormesh_expected(qm) -> dict:
    """
    Compute the expected ``pcolormesh`` series from the ``QuadMesh`` object.

    Uses the same ``quadmesh_to_logdata`` formula as the logging patch.
    """
    qm.figure.canvas.draw()
    V = qm.get_array()
    x_min, x_max = qm.sticky_edges.x[0], qm.sticky_edges.x[-1]
    y_min, y_max = qm.sticky_edges.y[0], qm.sticky_edges.y[-1]
    n_row, n_col = V.shape

    col_fracs = np.linspace(0.0, 1.0, n_col)
    row_fracs = np.linspace(0.0, 1.0, n_row)
    jj, ii    = np.meshgrid(col_fracs, row_fracs)

    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

    return {"plt_func": "pcolormesh",
            "x": xs.tolist(), "y": ys.tolist(), "z": zs.tolist()}


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib pcolormesh gallery example.

    Source (plt.style.use removed):
        x = [-3, -2, -1.6, ..., 2.3, 3]   (16 uneven x-vertices)
        X, Y = np.meshgrid(x, np.linspace(-3, 3, 128))
        Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
        ax.pcolormesh(X, Y, Z, vmin=-0.5, vmax=1.0)

    Produces 128 × 16 = 2048 logged points.
    sticky x extends beyond [-3, 3] because cells are centred on vertices.
    """
    x = [-3, -2, -1.6, -1.2, -.8, -.5, -.2, .1, .3, .5, .8, 1.1, 1.5, 1.9, 2.3, 3]
    X, Y = np.meshgrid(x, np.linspace(-3, 3, 128))
    Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

    qm = ax.pcolormesh(X, Y, Z, vmin=-0.5, vmax=1.0)

    return [_pcolormesh_expected(qm)]


def case_uniform_grid(ax):
    """Uniform x/y spacing — symmetric sticky-edge extension."""
    x = np.linspace(0, 4, 5)     # 5 vertices → 4 cells in x
    y = np.linspace(0, 3, 4)     # 4 vertices → 3 cells in y
    X, Y = np.meshgrid(x, y)
    Z = X + Y

    qm = ax.pcolormesh(X, Y, Z)

    return [_pcolormesh_expected(qm)]


def case_scalar_z_grid(ax):
    """pcolormesh with scalar x/y edges (non-meshgrid form)."""
    x = np.arange(0, 6)     # 6 x-edges → 5 x-cells
    y = np.arange(0, 4)     # 4 y-edges → 3 y-cells
    Z = np.random.default_rng(1).random((3, 5))

    qm = ax.pcolormesh(x, y, Z)

    return [_pcolormesh_expected(qm)]
