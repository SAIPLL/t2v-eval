"""
Test cases for ``DataFrame.plot.hexbin`` logging.

Strategy: extract from returned collection
-------------------------------------------
Identical to the matplotlib hexbin strategy.  Pandas delegates to
``ax.hexbin()``; the returned ``PolyCollection`` is the same object
the logging patch reads::

    logged x = collection.get_offsets()[:, 0]   (hex bin centres)
    logged y = collection.get_offsets()[:, 1]
    logged z = collection.get_array()            (counts or aggregated value)

Expected values are extracted from the same collection object.
"""

import numpy as np
import pandas as pd


def case_from_docs(ax):
    """
    Hexbin from the pandas docs:
        df = pd.DataFrame(np.random.randn(1000, 2), columns=["a","b"])
        df["b"] = df["b"] + np.arange(1000)
        df.plot.hexbin(x="a", y="b", gridsize=25)
    Using gridsize=10 for speed.
    """
    np.random.seed(123456)
    df = pd.DataFrame(np.random.randn(1000, 2), columns=["a", "b"])
    df["b"] = df["b"] + np.arange(1000)

    df.plot.hexbin(x="a", y="b", gridsize=10, ax=ax)
    # df.plot.hexbin returns Axes, not PolyCollection — get collection from ax
    ax.figure.canvas.draw()
    col = ax.collections[-1]
    xy  = col.get_offsets()
    V   = col.get_array()

    return [{"plt_func": "hexbin",
             "x": xy[:, 0].tolist(),
             "y": xy[:, 1].tolist(),
             "z": V.tolist()}]


def case_small_grid(ax):
    """Small gridsize — fewer bins, faster test."""
    np.random.seed(7)
    df = pd.DataFrame({"a": np.random.randn(200), "b": np.random.randn(200)})

    df.plot.hexbin(x="a", y="b", gridsize=5, ax=ax)
    ax.figure.canvas.draw()
    col = ax.collections[-1]
    xy  = col.get_offsets()
    V   = col.get_array()

    return [{"plt_func": "hexbin",
             "x": xy[:, 0].tolist(),
             "y": xy[:, 1].tolist(),
             "z": V.tolist()}]
