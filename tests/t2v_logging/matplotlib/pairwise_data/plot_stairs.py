"""
Test cases for ``Axes.stairs`` logging.

What is logged
--------------
``activate_axes_stairs`` extracts data from the ``StepPatch`` returned by
matplotlib via ``result.get_data()`` which yields ``(values, edges)``.
Each step is represented by its **centre x position** paired with its height:

    centers = (edges[:-1] + edges[1:]) / 2     # n values
    logged x = centers,  logged y = values      # vertical (default)
    logged x = values,   logged y = centers     # horizontal

For ``ax.stairs(values)`` with no explicit edges:
    matplotlib auto-generates  edges = [0, 1, 2, ..., n]
    so centers = [0.5, 1.5, ..., n-0.5]

Strategy: computed
------------------
All expected values are directly computable from the inputs.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib stairs gallery example.

    Source (plt.style.use removed):
        y = [4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0]
        ax.stairs(y, linewidth=2.5)

    Auto-generated edges = [0, 1, 2, ..., 8]
    Centers             = [0.5, 1.5, ..., 7.5]
    """
    y = np.array([4.8, 5.5, 3.5, 4.6, 6.5, 6.6, 2.6, 3.0])

    ax.stairs(y, linewidth=2.5)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    n       = len(y)
    edges   = np.arange(n + 1, dtype=float)        # [0, 1, ..., 8]
    centers = (edges[:-1] + edges[1:]) / 2         # [0.5, 1.5, ..., 7.5]

    return [{"plt_func": "stairs", "x": centers, "y": y}]


def case_explicit_edges(ax):
    """Stairs with explicit non-uniform edges."""
    values = np.array([2.0, 5.0, 3.0, 4.0])
    edges  = np.array([0.0, 1.0, 3.0, 4.0, 6.0])   # non-uniform bin widths

    ax.stairs(values, edges)

    centers = (edges[:-1] + edges[1:]) / 2          # [0.5, 2.0, 3.5, 5.0]

    return [{"plt_func": "stairs", "x": centers, "y": values}]


def case_horizontal(ax):
    """Stairs with ``orientation='horizontal'`` — x and y are swapped."""
    values = np.array([1.0, 3.0, 2.0])

    ax.stairs(values, orientation="horizontal")

    n       = len(values)
    edges   = np.arange(n + 1, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2          # [0.5, 1.5, 2.5]

    # horizontal: x = values, y = centers
    return [{"plt_func": "stairs", "x": values, "y": centers}]
