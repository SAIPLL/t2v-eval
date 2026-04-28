"""
Test cases for ``Axes3D.bar3d`` logging.

Strategy: computed
------------------
``activate_axes3D_bar3d`` logs the **top-corner** of each bar::

    logged x = np.array(x) + np.array(dx)
    logged y = np.array(y) + np.array(dy)
    logged z = np.array(z) + np.array(dz)

All three are directly computable from the inputs.  Unlike 2-D ``bar``
(which logs bar-centre via ``patches_to_logdata``), 3-D bar logs the
top corner explicitly in the logging patch itself.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib bar3d gallery example.

    Source (plt.style.use removed):
        x  = [1, 1, 2, 2]
        y  = [1, 2, 1, 2]
        z  = [0, 0, 0, 0]
        dx = np.ones_like(x) * 0.5
        dy = np.ones_like(x) * 0.5
        dz = [2, 3, 1, 4]
        ax.bar3d(x, y, z, dx, dy, dz)
    """
    x  = np.array([1, 1, 2, 2], dtype=float)
    y  = np.array([1, 2, 1, 2], dtype=float)
    z  = np.array([0, 0, 0, 0], dtype=float)
    dx = np.ones_like(x) * 0.5
    dy = np.ones_like(x) * 0.5
    dz = np.array([2, 3, 1, 4], dtype=float)

    ax.bar3d(x, y, z, dx, dy, dz)
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    return [{"plt_func": "bar3d",
             "x": x + dx, "y": y + dy, "z": z + dz}]


def case_non_zero_base(ax):
    """Bars starting at a non-zero z — logged z = z + dz, not just dz."""
    x  = np.array([0.0, 1.0, 2.0])
    y  = np.array([0.0, 0.0, 0.0])
    z  = np.array([1.0, 2.0, 0.5])   # non-zero base
    dx = np.array([0.8, 0.8, 0.8])
    dy = np.array([0.8, 0.8, 0.8])
    dz = np.array([3.0, 1.5, 2.0])

    ax.bar3d(x, y, z, dx, dy, dz)

    return [{"plt_func": "bar3d",
             "x": x + dx, "y": y + dy, "z": z + dz}]


def case_single_bar(ax):
    """Single bar — minimal case."""
    x, y, z   = [0.0], [0.0], [0.0]
    dx, dy, dz = [1.0], [1.0], [5.0]

    ax.bar3d(x, y, z, dx, dy, dz)

    return [{"plt_func": "bar3d",
             "x": np.array(x) + np.array(dx),
             "y": np.array(y) + np.array(dy),
             "z": np.array(z) + np.array(dz)}]
