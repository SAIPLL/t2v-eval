"""
Test cases for ``Axes3D.scatter`` (``scatter3d``) logging.

Strategy: extract from returned collection
-------------------------------------------
``activate_axes3D_scatter`` reads x/y/z from ``collection._offsets3d``.
The same source is used here to compute expected values, so expected and
logged are always consistent regardless of the random seed::

    x_data, y_data, z_data = collection._offsets3d

    logged plt_func = "scatter3d"
    logged x        = x_data
    logged y        = y_data
    logged z        = z_data

Note on the gallery example
----------------------------
The original code uses ``np.random.default_rng()`` (no seed), so the
coordinates change on every run.  The "extract from collection" strategy
makes the test pass regardless of the generated values.

For deterministic cases we use ``np.random.default_rng(seed=<n>)`` so that
failures reproduce consistently in CI.

Fixture note
------------
These cases receive a 3-D ``Axes3D`` object (not the default 2-D ``Axes``).
They must be registered in ``test_matplotlib_3d.py``, which creates the
correct projection.
"""

import numpy as np


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib 3D scatter gallery example.

    Source (plt.style.use and zticklabels removed — don't affect data):
        np.random.seed(19680801)
        n = 100
        rng = np.random.default_rng()
        xs = rng.uniform(23, 32, n)
        ys = rng.uniform(0, 100, n)
        zs = rng.uniform(-50, -25, n)
        ax.scatter(xs, ys, zs)

    ``np.random.default_rng()`` ignores the legacy seed, so coordinates are
    non-deterministic.  Expected values are extracted from ``_offsets3d``.
    """
    np.random.seed(19680801)         # legacy seed (ignored by default_rng)
    rng = np.random.default_rng()    # non-deterministic, matches gallery exactly
    xs = rng.uniform(23, 32, 100)
    ys = rng.uniform(0,  100, 100)
    zs = rng.uniform(-50, -25, 100)

    collection = ax.scatter(xs, ys, zs)
    ax.figure.canvas.draw()
    x_data, y_data, z_data = collection._offsets3d

    return [{"plt_func": "scatter3d",
             "x": x_data, "y": y_data, "z": z_data}]


def case_deterministic(ax):
    """
    Explicit known data — x/y/z expected values are directly computable.
    """
    rng = np.random.default_rng(seed=42)
    xs = rng.uniform(0, 5, 30)
    ys = rng.uniform(0, 5, 30)
    zs = rng.uniform(0, 5, 30)

    ax.scatter(xs, ys, zs)

    return [{"plt_func": "scatter3d", "x": xs, "y": ys, "z": zs}]


def case_with_size(ax):
    """3D scatter with explicit per-point marker size."""
    rng = np.random.default_rng(seed=7)
    xs = rng.uniform(0, 4, 20)
    ys = rng.uniform(0, 4, 20)
    zs = rng.uniform(0, 4, 20)
    s  = rng.uniform(10, 100, 20)

    collection = ax.scatter(xs, ys, zs, s=s)
    ax.figure.canvas.draw()
    x_data, y_data, z_data = collection._offsets3d
    s_data = collection.get_sizes()

    return [{"plt_func": "scatter3d",
             "x": x_data, "y": y_data, "z": z_data, "s": s_data}]
