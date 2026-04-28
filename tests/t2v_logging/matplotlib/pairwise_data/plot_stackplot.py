"""
Test cases for ``Axes.stackplot`` logging.

How stackplot is logged
------------------------
``stackplot`` is NOT directly monkey-patched.  Internally it calls
``fill_between`` once per data layer, which IS patched.  So each layer
produces one logged series with ``plt_func="fill_between"``.

For a stack with k data rows and n x-points:
  - k series are logged, one per layer
  - Each series has 2n+3 polygon vertices
  - Stack layers (bottom → top):
      layer 0:  fill_between(x, y1=zeros,   y2=stack[0])
      layer i:  fill_between(x, y1=stack[i-1], y2=stack[i])

Expected values use :func:`~tests.t2v_logging.helpers.fill_between_expected`
which applies the sorted-pair formula (same as fill_between tests).
"""

import numpy as np

from tests.t2v_logging.helpers import fill_between_expected


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib stackplot gallery example.

    Source (plt.style.use removed):
        x  = np.arange(0, 10, 2)
        ay = [1, 1.25, 2, 2.75, 3]
        by = [1, 1, 1, 1, 1]
        cy = [2, 1, 2, 1, 2]
        y  = np.vstack([ay, by, cy])
        ax.stackplot(x, y)

    Produces 3 fill_between series:
        layer 0: between 0          and ay
        layer 1: between ay         and ay+by
        layer 2: between ay+by      and ay+by+cy
    """
    x  = np.arange(0, 10, 2).astype(float)
    ay = np.array([1, 1.25, 2, 2.75, 3], dtype=float)
    by = np.array([1, 1,    1, 1,    1], dtype=float)
    cy = np.array([2, 1,    2, 1,    2], dtype=float)

    ax.stackplot(x, np.vstack([ay, by, cy]))
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    zeros  = np.zeros_like(x)
    stack0 = ay
    stack1 = ay + by
    stack2 = ay + by + cy

    return [
        fill_between_expected(x, zeros,  stack0),   # layer 0
        fill_between_expected(x, stack0, stack1),   # layer 1
        fill_between_expected(x, stack1, stack2),   # layer 2
    ]


def case_two_layers(ax):
    """Minimal two-layer stackplot — verifies layer count and boundary values."""
    x  = np.array([0.0, 1.0, 2.0, 3.0])
    y1 = np.array([1.0, 2.0, 1.0, 3.0])
    y2 = np.array([2.0, 1.0, 3.0, 1.0])

    ax.stackplot(x, np.vstack([y1, y2]))

    zeros  = np.zeros_like(x)
    stack0 = y1
    stack1 = y1 + y2

    return [
        fill_between_expected(x, zeros,  stack0),
        fill_between_expected(x, stack0, stack1),
    ]
