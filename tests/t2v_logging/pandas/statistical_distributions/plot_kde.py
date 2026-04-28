"""
Test cases for ``Series.plot.kde`` / ``DataFrame.plot.kde`` logging.

Strategy: partial check
------------------------
KDE delegates to ``ax.plot()`` after computing the density estimate
internally.  The logged plt_func is ``"plot"`` and the series contains
1 000 (x, y) grid points.

Because the exact KDE grid depends on pandas' internal bandwidth selection
and grid construction, we **do not check coordinate values**.  The test
verifies:
  - ``plt_func == "plot"``
  - series count == n_columns
  - x and y have the same length (length-consistency assertion)

If you want value-accuracy checks, compute the expected with scipy::

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data, bw_method="scott")   # pandas default
    x_grid = np.linspace(data.min(), data.max(), 1000)
    y_kde  = kde(x_grid)
"""

import numpy as np
import pandas as pd


def case_series_basic(ax):
    """Single Series KDE — one plot series."""
    np.random.seed(42)
    s = pd.Series(np.random.randn(100))
    s.plot.kde(ax=ax)
    # Only plt_func checked; x/y values omitted → length-consistency only
    return [{"plt_func": "plot"}]


def case_from_docs(ax):
    """
    KDE from the pandas docs:
        ser = pd.Series(np.random.randn(1000))
        ser.plot.kde()
    """
    np.random.seed(123456)
    ser = pd.Series(np.random.randn(1000))
    ser.plot.kde(ax=ax)
    return [{"plt_func": "plot"}]


def case_dataframe_two_columns(ax):
    """DataFrame KDE — one plot series per column."""
    np.random.seed(0)
    df = pd.DataFrame({"A": np.random.randn(50), "B": np.random.randn(50) + 2})
    df.plot.kde(ax=ax)
    # Two columns → two plot series
    return [{"plt_func": "plot"}, {"plt_func": "plot"}]
