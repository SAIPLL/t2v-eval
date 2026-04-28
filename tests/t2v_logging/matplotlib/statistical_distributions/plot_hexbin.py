"""
Test cases for ``Axes.hexbin`` logging.

Strategy: extract from returned collection
-------------------------------------------
``activate_axes_hexbin`` logs the hex bin centres and counts directly
from the ``PolyCollection`` that matplotlib's hexbin returns::

    xy = collection.get_offsets()   # shape (n_bins, 2) — bin centres
    V  = collection.get_array()     # shape (n_bins,)   — counts per bin

    logged x = xy[:, 0]
    logged y = xy[:, 1]
    logged z = V

Unlike ``hist2d``, there is no standalone numpy equivalent that replicates
the hexagonal binning geometry.  The expected values are therefore extracted
from the same ``PolyCollection`` object that the logging patch reads, making
expected and logged values identical by construction.

The case functions call ``ax.hexbin(...)`` which returns the collection, then
read the expected values from that collection before returning.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib hexbin gallery example.

    Source (plt.style.use removed):
        np.random.seed(1)
        x = np.random.randn(5000)
        y = 1.2 * x + np.random.randn(5000) / 3
        ax.hexbin(x, y, gridsize=20)
    """
    np.random.seed(1)
    x = np.random.randn(5000)
    y = 1.2 * x + np.random.randn(5000) / 3

    col = ax.hexbin(x, y, gridsize=20)
    ax.set(xlim=(-2, 2), ylim=(-3, 3))

    # Extract from the same source the logging patch reads
    ax.figure.canvas.draw()
    xy = col.get_offsets()
    V  = col.get_array()

    return [{"plt_func": "hexbin",
             "x": xy[:, 0].tolist(),
             "y": xy[:, 1].tolist(),
             "z": V.tolist()}]


def case_small_grid(ax):
    """Coarse hexbin grid — fewer bins, easier to reason about."""
    np.random.seed(42)
    x = np.random.uniform(-1, 1, 200)
    y = np.random.uniform(-1, 1, 200)

    col = ax.hexbin(x, y, gridsize=5)
    ax.figure.canvas.draw()
    xy = col.get_offsets()
    V  = col.get_array()

    return [{"plt_func": "hexbin",
             "x": xy[:, 0].tolist(),
             "y": xy[:, 1].tolist(),
             "z": V.tolist()}]


def case_with_extent(ax):
    """Hexbin with explicit extent — bins clipped to the given range."""
    np.random.seed(7)
    x = np.random.randn(1000)
    y = np.random.randn(1000)

    col = ax.hexbin(x, y, gridsize=10, extent=(-2, 2, -2, 2))
    ax.figure.canvas.draw()
    xy = col.get_offsets()
    V  = col.get_array()

    return [{"plt_func": "hexbin",
             "x": xy[:, 0].tolist(),
             "y": xy[:, 1].tolist(),
             "z": V.tolist()}]
