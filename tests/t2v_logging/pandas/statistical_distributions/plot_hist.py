"""
Test cases for ``Series.plot.hist`` / ``DataFrame.plot.hist`` logging.

Strategy: computed
------------------
Delegates to ``ax.hist()``.  Logged values (from ``patches_to_logdata``)
are the **top-centre** of each bar::

    logged_x = bin_centre   (= (edges[i] + edges[i+1]) / 2)
    logged_y = count

Expected values computed via ``numpy.histogram``.
One series per colour group (one colour per column when plotting a DataFrame
column-by-column, or one series when plotting a single Series).
"""

import numpy as np
import pandas as pd


def _hist_expected(data, bins, plt_func="hist") -> dict:
    """Compute expected (x=bin_centres, y=counts) for a single histogram."""
    counts, edges = np.histogram(data, bins=bins)
    centres       = (edges[:-1] + edges[1:]) / 2
    return {"plt_func": plt_func,
            "x": centres, "y": counts.astype(float)}


def case_series_basic(ax):
    """Single Series histogram — one logged series."""
    np.random.seed(42)
    s = pd.Series(np.random.randn(50))
    s.plot.hist(bins=5, ax=ax)
    return [_hist_expected(s.values, bins=5)]


def case_from_docs(ax):
    """
    Histogram from the pandas docs:
        df4["a"].plot.hist(orientation="horizontal", cumulative=True)
    Use a simpler vertical, non-cumulative call for clarity.
    """
    np.random.seed(123456)
    data = pd.Series(np.random.randn(1000) + 1)
    data.plot.hist(bins=10, ax=ax)
    return [_hist_expected(data.values, bins=10)]
