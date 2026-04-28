"""
Test cases for ``Axes.hist2d`` logging.

Strategy: computed
------------------
``activate_axes_hist2d`` calls ``quadmesh_to_logdata`` on the QuadMesh
returned by matplotlib.  The QuadMesh stores the 2-D count array ``V``
and the data extent via ``sticky_edges``.

``quadmesh_to_logdata`` then computes uniformly spaced grid centres::

    col_fracs = linspace(0, 1, n_col)   # n_col = n_x_bins
    row_fracs = linspace(0, 1, n_row)   # n_row = n_y_bins
    jj, ii    = meshgrid(col_fracs, row_fracs)
    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

Key points
----------
- matplotlib transposes the count array before passing to pcolormesh:
  ``V = counts.T``, so ``V.shape = (n_y_bins, n_x_bins)``.
- ``x_min, x_max`` = first and last x bin edge.
- ``y_min, y_max`` = first and last y bin edge.
- Total logged points = n_x_bins × n_y_bins.
- A single ``"hist2d"`` series is produced.
"""

import numpy as np


def _hist2d_expected(x_data, y_data, bins) -> dict:
    """
    Compute the expected ``hist2d`` series using ``numpy.histogram2d``.

    Parameters
    ----------
    x_data, y_data : array-like
        Input data arrays.
    bins : int, array, or pair of arrays
        Bin specification forwarded to ``numpy.histogram2d``.
    """
    counts, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=bins)

    # matplotlib transposes the count matrix before creating the QuadMesh
    V = counts.T
    n_row, n_col = V.shape
    x_min, x_max = float(x_edges[0]),  float(x_edges[-1])
    y_min, y_max = float(y_edges[0]),  float(y_edges[-1])

    col_fracs = np.linspace(0.0, 1.0, n_col)
    row_fracs = np.linspace(0.0, 1.0, n_row)
    jj, ii    = np.meshgrid(col_fracs, row_fracs)

    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

    return {
        "plt_func": "hist2d",
        "x": xs.tolist(),
        "y": ys.tolist(),
        "z": zs.tolist(),
    }


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib hist2d gallery example.

    Source (plt.style.use removed):
        np.random.seed(1)
        x = np.random.randn(5000)
        y = 1.2 * x + np.random.randn(5000) / 3
        ax.hist2d(x, y, bins=(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1)))

    Produces a 59×59 = 3481-point series.
    """
    np.random.seed(1)
    x = np.random.randn(5000)
    y = 1.2 * x + np.random.randn(5000) / 3

    x_bins = np.arange(-3, 3, 0.1)
    y_bins = np.arange(-3, 3, 0.1)

    ax.hist2d(x, y, bins=(x_bins, y_bins))
    ax.set(xlim=(-2, 2), ylim=(-3, 3))

    return [_hist2d_expected(x, y, bins=(x_bins, y_bins))]


def case_integer_bins(ax):
    """Scalar bin count — let numpy decide the edges."""
    np.random.seed(7)
    x = np.random.normal(0, 1, 500)
    y = np.random.normal(0, 1, 500)

    ax.hist2d(x, y, bins=10)

    return [_hist2d_expected(x, y, bins=10)]


def case_asymmetric_bins(ax):
    """Different bin counts for x and y axes."""
    np.random.seed(3)
    x = np.random.uniform(-2, 2, 300)
    y = np.random.uniform(0,  5, 300)

    ax.hist2d(x, y, bins=(8, 12))

    return [_hist2d_expected(x, y, bins=(8, 12))]
