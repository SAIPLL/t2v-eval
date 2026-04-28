"""
Test cases for ``Axes.eventplot`` logging.

Strategy: computed
------------------
``activate_axes_eventplot`` merges all event groups into a single
``"scatter"`` series.  The midpoint of each tick segment gives one
logged (x, y) point.  The orientation determines which axis holds the
event values and which holds the lineoffset:

    orientation="vertical"   (events on y-axis)
        x = [lineoffset] × n_events,   y = sorted(events)

    orientation="horizontal"  (events on x-axis, matplotlib default)
        x = sorted(events),            y = [lineoffset] × n_events

All groups are merged in lineoffsets order.  matplotlib sorts events
within each EventCollection by default (ascending).

Logged structure (one series total)
-------------------------------------
    plt_func = "scatter"
    x        = group₀_x  +  group₁_x  +  …
    y        = group₀_y  +  group₁_y  +  …
"""

import numpy as np


def _eventplot_expected(D, lineoffsets, orientation="horizontal") -> dict:
    """
    Compute the expected merged scatter series for an eventplot call.

    Parameters
    ----------
    D : array-like of shape (n_groups, n_events)
        Event data — one row per event group.
    lineoffsets : list of float
        Position of each group on the non-event axis.
    orientation : {"horizontal", "vertical"}
        Matches the ``orientation`` kwarg passed to ``ax.eventplot``.
    """
    x, y = [], []
    for col, pos in zip(D, lineoffsets):
        n = len(col)
        if orientation == "vertical":
            x.extend([float(pos)] * n)
            y.extend(sorted(col))
        else:                               # horizontal (matplotlib default)
            x.extend(sorted(col))
            y.extend([float(pos)] * n)
    return {"plt_func": "scatter", "x": x, "y": y}


def case_from_gallery(ax):
    """
    Reproduction of the matplotlib eventplot gallery example.

    Source (plt.style.use removed):
        np.random.seed(1)
        x = [2, 4, 6]
        D = np.random.gamma(4, size=(3, 50))
        ax.eventplot(D, orientation="vertical", lineoffsets=x, linewidth=0.75)

    50 events per group → 150-point scatter series.
    y = sorted events (values on y-axis), x = lineoffset (fixed per group).
    """
    np.random.seed(1)
    x = [2, 4, 6]
    D = np.random.gamma(4, size=(3, 50))

    ax.eventplot(D, orientation="vertical", lineoffsets=x, linewidth=0.75)
    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    return [_eventplot_expected(D, x, orientation="vertical")]


def case_horizontal(ax):
    """
    Horizontal eventplot — events along x-axis, offsets along y-axis.

    x = sorted event values,   y = lineoffset (constant per group).
    """
    np.random.seed(5)
    offsets = [1.0, 3.0]
    D = np.random.uniform(0, 10, size=(2, 20))

    ax.eventplot(D, orientation="horizontal", lineoffsets=offsets)

    return [_eventplot_expected(D, offsets, orientation="horizontal")]


def case_single_group(ax):
    """
    Single event group with default orientation (horizontal).

    x = sorted events,   y = [lineoffset] × n_events.
    """
    np.random.seed(0)
    events = np.random.exponential(2, size=30)

    ax.eventplot([events], lineoffsets=[3.0])   # default: horizontal

    return [_eventplot_expected([events], [3.0], orientation="horizontal")]
