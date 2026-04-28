"""
Logging utilities for T2V evaluation.

Provides:
- Log-directory lifecycle management (:func:`setup_log_dir`, :func:`reset_log_dir`)
- A simple JSON-backed flag store (:func:`get_variable`, :func:`set_variable`)
- A JSON encoder that handles NumPy / Pandas types (:class:`JSONNumberEncoder`)
- Artist-to-log-data converters (``*_to_logdata`` functions) consumed by the
  matplotlib monkey-patches in ``t2v_eval.logging.matplotlib_logging``
"""

import datetime
import json
import os
import shutil
import time
from collections import defaultdict
from collections.abc import KeysView, ValuesView
from contextlib import contextmanager

import matplotlib
import matplotlib.dates
import numpy as np
import pandas as pd

from t2v_eval import __version__ as T2V_EVAL_VERSION


# ---------------------------------------------------------------------------
# Log-directory management
# ---------------------------------------------------------------------------

LOG_DIR = "/tmp/t2vlog"


def reset_log_dir() -> None:
    """Reset the log directory to the default path ``/tmp/t2vlog``."""
    setup_log_dir("/tmp/t2vlog")


def setup_log_dir(log_dir: str) -> None:
    """
    Set the active log directory, removing any existing content.

    Parameters
    ----------
    log_dir : str
        Absolute path to use as the logging root.
    """
    global LOG_DIR
    LOG_DIR = log_dir
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Simple JSON-backed flag store
# ---------------------------------------------------------------------------

def get_variable(variable_name: str):
    """
    Read *variable_name* from ``variables.json`` in the active log directory.

    Returns ``None`` if the file does not exist or the key is absent.
    """
    variable_path = os.path.join(LOG_DIR, "variables.json")
    if not os.path.exists(variable_path):
        return None
    with open(variable_path, encoding="utf-8") as fh:
        return json.load(fh).get(variable_name)


def set_variable(variable_name: str, value) -> None:
    """
    Write ``{variable_name: value}`` to ``variables.json`` in the log directory.

    The file is overwritten on every call (only one variable is stored at a time).
    """
    variable_path = os.path.join(LOG_DIR, "variables.json")
    with open(variable_path, "w", encoding="utf-8") as fh:
        json.dump({variable_name: value}, fh, indent=4, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Logging pause context manager
# ---------------------------------------------------------------------------

@contextmanager
def logging_paused():
    """
    Context manager: pause T2V logging for the duration of the block.

    Yields the *previous* value of ``T2V_ISLOG`` so callers can check whether
    logging was active before the pause::

        with logging_paused() as was_logging:
            result = original_method(...)
        if was_logging:
            ...log result...
    """
    was_logging = get_variable("T2V_ISLOG")
    set_variable("T2V_ISLOG", False)
    try:
        yield was_logging
    finally:
        set_variable("T2V_ISLOG", was_logging)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------

class JSONNumberEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy / Pandas / Python numeric types."""

    def default(self, obj):
        try:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.ma.MaskedArray):
                # Fill masked values with the array minimum before serialising
                return obj.filled(np.min(obj.compressed())).tolist()
            if isinstance(obj, complex):
                return float(obj.real)
            if isinstance(obj, (KeysView, ValuesView)):
                return list(obj)
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, pd.Index):
                return obj.tolist()
            if isinstance(obj, pd.Categorical):
                return obj.tolist()
            if isinstance(obj, range):
                return list(obj)
            return super().default(obj)
        except Exception:
            print(f"Error encoding object: {obj!r}, type: {type(obj)}")
            s = str(obj)
            return None if (s.startswith("<") and s.endswith(">")) else s


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

# Types whose presence in a data array indicates datetime values
_DATETIME_TYPES = (
    matplotlib.dates.DateFormatter,
    pd.Timestamp,
    pd.DatetimeIndex,
    pd.Period,
    np.datetime64,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
)


def is_datetime(items: list) -> bool:
    """Return True if every item in *items* is a parseable datetime string."""
    if not items:
        return False
    for item in items:
        if not isinstance(item, str):
            return False
        try:
            pd.to_datetime(item)
        except ValueError:
            return False
    return True


def is_datetime_line(x_data: list) -> bool:
    """
    Return True if the first element of *x_data* is a datetime-like object.

    *x_data* is expected to come from ``Line2D.get_xdata()``.
    """
    return len(x_data) > 0 and isinstance(x_data[0], _DATETIME_TYPES)


# ---------------------------------------------------------------------------
# Core log writer
# ---------------------------------------------------------------------------

def log_data(ax, plt_func: str, args, kargs: dict, data_series: list) -> None:
    """
    Serialise one plotting call to a timestamped JSON file in ``LOG_DIR``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which the plot was drawn.
    plt_func : str
        Name of the matplotlib function (e.g. ``"plot"``, ``"scatter"``).
    args : sequence
        Positional arguments originally passed to the plotting function.
    kargs : dict
        Keyword arguments originally passed to the plotting function.
    data_series : list
        List of data-series dicts extracted from the matplotlib artists.
    """
    ax_id   = id(ax)
    fig_id  = id(ax.figure)
    ts      = str(time.time()).replace(".", "")
    out     = os.path.join(LOG_DIR, f"{fig_id}_{ax_id}_{ts}.json")

    save_kargs = kargs.copy()
    try:
        save_kargs["ax_col"] = ax.get_subplotspec().colspan.start
        save_kargs["ax_row"] = ax.get_subplotspec().rowspan.start
    except Exception as e:
        print(f"Error getting ax_col / ax_row: {e}")
        save_kargs["ax_col"] = 0
        save_kargs["ax_row"] = 0

    # Drop any series keys that cannot be JSON-serialised
    for series in data_series:
        for key in list(series.keys()):
            try:
                json.dumps(series[key], cls=JSONNumberEncoder,
                           indent=4, ensure_ascii=False)
            except Exception:
                assert key not in ("x", "y", "z", "V", "labels"), \
                    f"Critical key {key!r} failed JSON serialisation."
                series[key] = None

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "t2v_eval_version": T2V_EVAL_VERSION,
                "plt_func":         plt_func,
                "args":             args,
                "kargs":            save_kargs,
                "data_series":      data_series,
            },
            fh,
            indent=4,
            ensure_ascii=False,
            cls=JSONNumberEncoder,
        )


# ---------------------------------------------------------------------------
# Artist → log-data converters
# ---------------------------------------------------------------------------

def line2d_to_logdata(line2d,
                      is_x_datetime: bool = False,
                      is_y_datetime: bool = False) -> list:
    """
    Convert a ``Line2D`` to a one-element data-series list.

    Parameters
    ----------
    line2d : matplotlib.lines.Line2D
    is_x_datetime / is_y_datetime : bool
        Datetime hints from the parent axes; supplemented by inspecting the
        raw data directly.
    """
    x_raw = line2d.get_xdata(orig=True)
    y_raw = line2d.get_ydata(orig=True)
    is_x_datetime = is_x_datetime or is_datetime_line(x_raw)
    is_y_datetime = is_y_datetime or is_datetime_line(y_raw)

    return [{
        "x":             line2d.get_xdata(orig=False),
        "y":             line2d.get_ydata(orig=False),
        "color":         line2d.get_color(),
        "linestyle":     line2d.get_linestyle(),
        "linewidth":     line2d.get_linewidth(),
        "marker":        line2d.get_marker(),
        "markersize":    line2d.get_markersize(),
        "label":         line2d.get_label(),
        "is_x_datetime": is_x_datetime,
        "is_y_datetime": is_y_datetime,
    }]


def linecollection_to_logdata(linecollection) -> list:
    """
    Convert a ``LineCollection`` to a list of per-segment data-series dicts.

    Segments with fewer than two points are skipped.
    """
    result   = []
    segments = linecollection.get_segments()
    for seg in segments:
        if seg.shape[0] < 2:
            continue
        result.append({
            "x":         seg[:, 0],
            "y":         seg[:, 1],
            "color":     linecollection.get_color()[0],
            "linewidth": linecollection.get_linewidth()[0],
        })
    return result


def eventcollection_to_logdata(eventcollection) -> list:
    """
    Convert an ``EventCollection`` to a one-element data-series list.

    Each segment's midpoint is used as the representative (x, y) point.
    """
    xs, ys = [], []
    for seg in eventcollection.get_segments():
        mid = np.mean(seg, axis=0)
        xs.append(mid[0])
        ys.append(mid[1])
    return [{
        "x":     xs,
        "y":     ys,
        "color": eventcollection.get_color()[0],
    }]


def pathcollection_to_logdata(pathcollection) -> list:
    """
    Convert a ``PathCollection`` (e.g. from ``scatter``) to a data-series list.

    Single-entry colour / size arrays are broadcast to match the point count.
    Arrays whose length differs from the point count are dropped (set to
    ``None``) so that downstream evaluation code always sees consistent lengths.
    """
    x, y   = pathcollection.get_offsets().T
    colors = pathcollection.get_facecolor()
    sizes  = pathcollection.get_sizes()
    n      = len(x)

    if   len(colors) == 1: colors = np.full((n, 4), colors[0])
    elif len(colors) != n: colors = None   # length mismatch — drop

    if   len(sizes)  == 1: sizes  = np.full((n,), sizes[0])
    elif len(sizes)  != n: sizes  = None   # length mismatch — drop

    return [{"x": x, "y": y, "color": colors, "s": sizes}]


def polycollection_to_logdata(poly) -> list:
    """
    Extract x/y vertices, face colour, and alpha from a ``PolyCollection``.

    Returns a one-element data-series list suitable for :func:`log_data`.
    Used by ``fill_between`` and ``fill_betweenx`` patches.
    """
    paths  = poly.get_paths()
    colors = poly.get_facecolor()
    x, y   = [], []
    for path in paths:
        v = path.vertices
        x.extend(v[:, 0])
        y.extend(v[:, 1])
    return [{
        "x":     x,
        "y":     y,
        "color": colors[0] if len(colors) > 0 else None,
        "alpha": poly.get_alpha(),
    }]


def patchcollection_to_logdata(collection) -> list:
    """
    Convert a ``PatchCollection`` (e.g. from geopandas) to a data-series list.

    Each path becomes one series entry with x/y border coordinates, a z value
    (the scalar array value), and the face colour.
    """
    paths      = collection.get_paths()
    array      = collection.get_array()
    facecolors = collection.get_facecolors()

    if array is None:
        array = [np.nan] * len(paths)
    if len(array) == 1 and len(paths) > 1:
        array = [array[0]] * len(paths)
    if len(facecolors) == 0:
        facecolors = np.zeros((len(paths), 4))   # fully transparent
    if len(facecolors) == 1 and len(paths) > 1:
        facecolors = [facecolors[0]] * len(paths)

    arr_min, arr_max = np.min(array), np.max(array)
    data_series = []
    for path, value, facecolor in zip(paths, array, facecolors):
        x = path.vertices[:, 0]
        y = path.vertices[:, 1]
        alpha = (value - arr_min) / (arr_max - arr_min) if arr_max != arr_min else 0.0
        data_series.append({
            "x":     x.tolist(),
            "y":     y.tolist(),
            "z":     (value * np.ones_like(x)).tolist(),
            "color": facecolor.tolist(),
            "alpha": alpha,
        })
    return data_series


def patches_to_logdata(patches, orientation: str = "vertical",
                       comp_name: str = "") -> list:
    """
    Convert a list of ``Patch`` objects (from ``bar`` / ``barh``) to log data.

    Patches of the same face colour are merged into a single series.  The
    representative point for each patch is the top-centre (vertical bars) or
    right-centre (horizontal bars).

    Parameters
    ----------
    patches : list of matplotlib.patches.Patch
    orientation : {"vertical", "horizontal"}
    comp_name : str
        Label suffix stored in each series dict.
    """
    by_color: dict = defaultdict(lambda: defaultdict(list))
    for patch in patches:
        x, y   = patch.get_x(), patch.get_y()
        w, h   = patch.get_width(), patch.get_height()
        color  = patch.get_facecolor()
        if orientation == "vertical":
            px, py = x + w / 2, y + h
        else:
            px, py = x + w,     y + h / 2
        by_color[color]["x"].append(px)
        by_color[color]["y"].append(py)
        by_color[color]["color"].append(color)
        by_color[color]["orientation"] = orientation
        by_color[color]["comp_name"]   = f"_{comp_name}_patches"
    return list(by_color.values())


def errorbarcontainer_to_logdata(errorbar_ctn, comp_name: str = "") -> list:
    """
    Convert an ``ErrorbarContainer`` to a data-series list.

    The data line (if present) and bar-line collections are included;
    cap lines are intentionally skipped.

    Parameters
    ----------
    errorbar_ctn : matplotlib.container.ErrorbarContainer
    comp_name : str
        Label suffix stored in each series dict.
    """
    data_series = []

    data_line = errorbar_ctn[0]
    if data_line is not None:
        data_series += line2d_to_logdata(data_line)
        data_series[-1]["comp_name"] = f"_{comp_name}_errorbar_data_line"

    # caplines = errorbar_ctn[1]  # skipped intentionally

    for barlinecol in errorbar_ctn[2]:
        for s in linecollection_to_logdata(barlinecol):
            s["comp_name"] = f"_{comp_name}_errorbar_barlinecol"
            data_series.append(s)

    return data_series


def get_mid_points_of_linecollection(line_collection) -> list:
    """
    Return the midpoint and colour of each segment in *line_collection*.

    Returns an empty list when *line_collection* is ``None``.

    Returns
    -------
    list of (x, y, color) tuples
    """
    if line_collection is None:
        return []

    segments = line_collection.get_segments()
    colors   = line_collection.get_color()
    if colors.shape[0] == 1:
        colors = np.full((len(segments), 4), colors[0])

    return [
        (np.mean(seg, axis=0)[0], np.mean(seg, axis=0)[1], color)
        for seg, color in zip(segments, colors)
    ]


def quadmesh_to_logdata(quadmesh) -> list:
    """
    Convert a ``QuadMesh`` to a flat data-series list of (x, y, z) points.

    Grid cell positions are computed by linear interpolation between the mesh
    sticky edges.  The full value array ``V`` is also stored for reference.
    """
    V = quadmesh.get_array()
    x_min, x_max = quadmesh.sticky_edges.x
    y_min, y_max = quadmesh.sticky_edges.y
    n_row, n_col = V.shape

    col_fracs = np.linspace(0.0, 1.0, n_col) if n_col > 1 else np.zeros(n_col)
    row_fracs = np.linspace(0.0, 1.0, n_row) if n_row > 1 else np.zeros(n_row)
    jj, ii = np.meshgrid(col_fracs, row_fracs)

    xs = (x_min + (x_max - x_min) * jj).ravel()
    ys = (y_min + (y_max - y_min) * ii).ravel()
    zs = V.ravel()

    return [{
        "V":    V,
        "x":    xs.tolist(),
        "y":    ys.tolist(),
        "z":    zs.tolist(),
        "cmin": float(np.min(V)),
        "cmax": float(np.max(V)),
    }]


def contours_to_logdata(contours) -> list:
    """
    Convert a ``ContourSet`` to a per-segment data-series list.

    Each contour segment is stored with its z level and the global min/max
    of all contour levels.
    """
    data_series = []
    levels = contours.levels
    cmin, cmax = float(np.min(levels)), float(np.max(levels))
    for layer, segments in zip(contours.layers, contours.allsegs):
        for seg in segments:
            data_series.append({
                "x":    seg[:, 0],
                "y":    seg[:, 1],
                "z":    np.full(len(seg), layer),
                "cmin": cmin,
                "cmax": cmax,
            })
    return data_series
