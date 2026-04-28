"""
Test cases for ``Axes.scatter`` logging.

Each ``case_*`` function:
  1. Receives a fresh ``matplotlib.axes.Axes`` object.
  2. Calls one or more plot commands (T2V logging must already be active).
  3. Returns a list of expected-series dicts — one dict per series that
     should appear in the log.

Expected-series dict keys
-------------------------
plt_func : str
    The ``plt_func`` value that must appear in the log (required).
x, y : list of float
    Expected coordinate values checked with ``np.testing.assert_allclose``.
z, s, t : list of float, optional
    Additional coordinates to verify; omit keys you do not want checked.

The test runner (``test_matplotlib_pairwise.py``) automatically collects
every ``case_*`` function defined here.
"""


def case_basic(ax):
    """Plain scatter — x and y only."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    ax.scatter(x, y)
    return [{"plt_func": "scatter", "x": x, "y": y}]


def case_with_size_array(ax):
    """Scatter with a per-point size array — s must be logged at full length."""
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0, 6.0]
    s = [10.0, 50.0, 200.0]
    ax.scatter(x, y, s=s)
    return [{"plt_func": "scatter", "x": x, "y": y, "s": s}]


def case_with_single_size(ax):
    """Scatter with a scalar size — should be broadcast to match len(x)."""
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 1.0, 1.0, 1.0]
    ax.scatter(x, y, s=100)
    return [{"plt_func": "scatter", "x": x, "y": y, "s": [100.0] * len(x)}]


def case_multiple_calls(ax):
    """Two separate scatter calls — should produce two logged series."""
    x1, y1 = [1.0, 2.0], [3.0, 4.0]
    x2, y2 = [5.0, 6.0], [7.0, 8.0]
    ax.scatter(x1, y1)
    ax.scatter(x2, y2)
    return [
        {"plt_func": "scatter", "x": x1, "y": y1},
        {"plt_func": "scatter", "x": x2, "y": y2},
    ]
