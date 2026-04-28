"""
Shared helper utilities for T2V logging tests.

Provides:
- :func:`read_log_series` — load all logged data-series from a log directory
- :func:`load_golden` — load expected series from a pre-verified golden log dir
- :func:`assert_lengths_consistent` — verify x/y/z/s/t all have equal length
- :func:`assert_series_matches` — compare a logged series against expected values
- :func:`assert_series_set_matches` — order-independent version of the above
- :class:`UnorderedSeries` — sentinel list signalling order-independent matching

Three strategies for expected values
--------------------------------------
**Computed (ordered)** — the case function returns a plain ``list[dict]``.
Series are matched positionally (index 0 vs index 0, etc.).
Use when the logging order is deterministic.

**Computed (unordered)** — the case function returns an :class:`UnorderedSeries`
(a ``list`` subclass).  The test runner sorts both logged and expected by a
composite key before matching pair-wise.  Use when seaborn controls the drawing
order of hue groups or facets and the order is not predictable from the inputs
(e.g. ``histplot`` with ``hue=``, ``ecdfplot`` with ``hue=``).

**Golden log dir** — the case function returns a ``str`` or ``pathlib.Path``.
"""

import glob
import json
import os
from pathlib import Path

import numpy as np

# Log files with these name prefixes are metadata, not data series
_RESERVED_PREFIXES = ("variables", "evaluation", "execution")

# Coordinate keys checked for length consistency and value accuracy
_COORD_KEYS = ("x", "y", "z", "s", "t")


# ---------------------------------------------------------------------------
# Log readers
# ---------------------------------------------------------------------------

def read_log_series(log_dir: str, plt_func: str = None) -> list:
    """
    Read all data-series from every JSON log file in *log_dir*.

    Parameters
    ----------
    log_dir : str
        Path to a T2V log directory.
    plt_func : str, optional
        When given, only series whose ``plt_func`` matches are returned.

    Returns
    -------
    list of dict
        Flat list of series dicts, each augmented with a ``"plt_func"`` key
        taken from the parent log file.
    """
    series = []
    for path in sorted(glob.glob(os.path.join(str(log_dir), "*.json"))):
        stem = os.path.basename(path).rsplit(".", 1)[0]
        if stem.startswith(_RESERVED_PREFIXES):
            continue
        with open(path, encoding="utf-8") as fh:
            log = json.load(fh)
        func = log.get("plt_func", "")
        if plt_func is not None and func != plt_func:
            continue
        for s in log.get("data_series", []):
            series.append({"plt_func": func, **s})
    return series


def load_golden(golden_dir) -> list:
    """
    Load expected series from a manually verified golden log directory.

    Use this when the transform from plot inputs to logged values is
    non-trivial and cannot be easily computed.  The golden directory should
    be committed to the repository alongside the test.

    Parameters
    ----------
    golden_dir : str or pathlib.Path
        Path to the golden T2V log directory.

    Returns
    -------
    list of dict
        Same format as :func:`read_log_series`.
    """
    return read_log_series(str(golden_dir))


def resolve_expected(case_result) -> list:
    """
    Resolve the return value of a case function into a list of expected dicts.

    - If *case_result* is a ``list`` → returned as-is (computed expected).
    - If *case_result* is a ``str`` or ``Path`` → treated as a golden log dir
      and loaded via :func:`load_golden`.

    Parameters
    ----------
    case_result : list or str or pathlib.Path
        Direct return value from a ``case_*`` function.
    """
    if isinstance(case_result, (str, Path)):
        return load_golden(case_result)
    return case_result


# ---------------------------------------------------------------------------
# Plot-type expected-value builders
# ---------------------------------------------------------------------------

def fill_betweenx_expected(y, x1, x2) -> dict:
    """
    Build the expected-series dict for ``fill_betweenx(y, x1, x2)``.

    Identical multiset formula to :func:`fill_between_expected` but with
    the axes roles swapped — ``y`` is the shared axis, ``x1``/``x2`` are
    the two boundaries::

        exp_x = list(x1) + list(x2) + [x2[-1], x2[0], x2[0]]
        exp_y = list(y)  + list(y)  + [y[-1],  y[0],  y[0] ]

    Parameters
    ----------
    y : array-like
        Shared y coordinates.
    x1, x2 : array-like
        Left and right boundary curves.
    """
    y, x1, x2 = list(y), list(x1), list(x2)
    return {
        "plt_func": "fill_betweenx",
        "x": x1 + x2 + [x2[-1], x2[0], x2[0]],
        "y": y  + y  + [y[-1],  y[0],  y[0] ],
        "_sort": True,
    }


def fill_between_expected(x, y1, y2) -> dict:
    """
    Build the expected-series dict for ``fill_between(x, y1, y2)``.

    Uses ``"_sort": True`` because the polygon vertex order does not match the
    input order.  The multiset formula is::

        exp_x = list(x)  + list(x)  + [x[-1],   x[0],   x[0]]
        exp_y = list(y1) + list(y2) + [y2[-1], y2[0], y2[0]]

    This produces the same sorted ``(x, y)`` pairs as the 2n+3 polygon vertices
    logged by ``polycollection_to_logdata``.

    Parameters
    ----------
    x : array-like
        Shared x coordinates.
    y1, y2 : array-like
        The two boundary curves passed to ``fill_between``.

    Returns
    -------
    dict
        Expected-series dict ready for :func:`assert_series_matches`.
    """
    x, y1, y2 = list(x), list(y1), list(y2)
    return {
        "plt_func": "fill_between",
        "x": x + x + [x[-1], x[0], x[0]],
        "y": y1 + y2 + [y2[-1], y2[0], y2[0]],
        "_sort": True,
    }


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_lengths_consistent(series: dict) -> None:
    """
    Assert that all coordinate arrays present in *series* have the same length.

    Checks ``x``, ``y``, ``z``, ``s``, and ``t``.  Keys that are absent or
    ``None`` are skipped.

    Parameters
    ----------
    series : dict
        A single logged data-series dict (as returned by :func:`read_log_series`).
    """
    lengths = {
        key: len(series[key])
        for key in _COORD_KEYS
        if series.get(key) is not None
    }
    unique = set(lengths.values())
    assert len(unique) <= 1, (
        f"Length mismatch in series (plt_func={series.get('plt_func')!r}): "
        + ", ".join(f"{k}={v}" for k, v in sorted(lengths.items()))
    )


def assert_series_matches(logged: dict, expected: dict,
                           atol: float = 1e-4) -> None:
    """
    Assert that *logged* matches *expected* on plt_func and coordinate values.

    Only keys present (and not ``None``) in *expected* are checked.  This lets
    callers omit keys they do not want to verify (e.g. ``color``).
    Coordinate keys missing from *expected* are silently skipped — use this
    to do a partial check (plt_func + length only, no value accuracy).

    Set ``"_sort": True`` in *expected* to sort ``(x, y)`` pairs before
    comparing — useful when the logged vertex order is not predictable (e.g.
    polygon outlines from ``fill_between``).

    Parameters
    ----------
    logged : dict
        A single logged data-series dict.
    expected : dict
        Expected values.  Must contain ``"plt_func"``; may contain any subset
        of ``"x"``, ``"y"``, ``"z"``, ``"s"``, ``"t"``.
        Optional ``"_sort": True`` sorts ``(x, y)`` pairs before comparing.
    atol : float
        Absolute tolerance passed to ``numpy.testing.assert_allclose``.
    """
    assert logged["plt_func"] == expected["plt_func"], (
        f"plt_func mismatch: logged {logged['plt_func']!r}, "
        f"expected {expected['plt_func']!r}"
    )

    if expected.get("_sort"):
        _assert_xy_sorted(logged, expected, atol)
        return

    for key in _COORD_KEYS:
        exp_val = expected.get(key)
        if exp_val is None:
            continue
        assert key in logged and logged[key] is not None, (
            f"Key {key!r} expected but missing or None in logged series "
            f"(plt_func={logged['plt_func']!r})."
        )
        np.testing.assert_allclose(
            logged[key], exp_val, atol=atol,
            err_msg=f"Value mismatch for key {key!r} "
                    f"(plt_func={logged['plt_func']!r})",
        )


def _assert_xy_sorted(logged: dict, expected: dict,
                       atol: float = 1e-4) -> None:
    """
    Sort ``(x, y)`` pairs from both *logged* and *expected*, then compare.

    Used when the polygon vertex order is undefined (e.g. ``fill_between``).
    Both sequences must have the same length after sorting.

    Parameters
    ----------
    logged, expected : dict
        Series dicts.  Both must contain ``"x"`` and ``"y"``.
    atol : float
        Absolute tolerance for each coordinate.
    """
    logged_pairs   = sorted(zip(logged["x"],   logged["y"]))
    expected_pairs = sorted(zip(expected["x"], expected["y"]))

    assert len(logged_pairs) == len(expected_pairs), (
        f"Sorted pair count mismatch (plt_func={logged['plt_func']!r}): "
        f"logged {len(logged_pairs)}, expected {len(expected_pairs)}"
    )

    logged_x,   logged_y   = zip(*logged_pairs)
    expected_x, expected_y = zip(*expected_pairs)

    np.testing.assert_allclose(
        logged_x, expected_x, atol=atol,
        err_msg=f"Sorted x mismatch (plt_func={logged['plt_func']!r})",
    )
    np.testing.assert_allclose(
        logged_y, expected_y, atol=atol,
        err_msg=f"Sorted y mismatch (plt_func={logged['plt_func']!r})",
    )


# ---------------------------------------------------------------------------
# Order-independent series matching
# ---------------------------------------------------------------------------

class UnorderedSeries(list):
    """
    Sentinel list returned by case functions when the seaborn drawing order
    of hue groups or facets is not predictable from the input data.

    The test runner detects this type and uses :func:`assert_series_set_matches`
    instead of positional zip-matching.
    """


def _series_sort_key(s: dict) -> tuple:
    """
    Composite sort key for a single series dict.

    Primary key: ``True`` for non-empty series, ``False`` for empty.
    This ensures empty legend artefacts (n=0) always sort before data series,
    preventing collision with data series that share the same x position.

    Secondary / tertiary keys: median finite x and y values so that:
    - For histograms with shared bin centres (same x), y median differs → sorted.
    - For ECDF / KDE with per-group x ranges, x median differs → sorted.
    - ``-inf`` / NaN values (ECDF start, artefacts) are filtered before median.
    """
    x_raw = s.get("x", [])
    y_raw = s.get("y", [])
    # Filter non-finite / NaN values
    x_fin = [v for v in x_raw
             if v is not None and np.isfinite(float(v))]
    y_fin = [v for v in y_raw
             if v is not None and np.isfinite(float(v))]
    nonempty = len(x_fin) > 0          # False → empty artefact sorts first
    xm = float(x_fin[len(x_fin) // 2]) if x_fin else 0.0
    ym = float(y_fin[len(y_fin) // 2]) if y_fin else 0.0
    return (nonempty, round(xm, 6), round(ym, 6))


def assert_series_set_matches(logged_list: list, expected_list: list,
                               atol: float = 1e-4) -> None:
    """
    Order-independent series matching.

    Both lists are sorted by :func:`_series_sort_key`, then compared
    pair-wise with :func:`assert_series_matches` and
    :func:`assert_lengths_consistent`.

    Use this when the seaborn drawing order of hue groups or facets is
    non-deterministic from the perspective of the test (e.g. seaborn sorts
    categories internally in a way that depends on the data).

    Parameters
    ----------
    logged_list : list of dict
        All series read from the log directory.
    expected_list : list of dict
        Expected series dicts (values checked where present, skipped if None).
    atol : float
        Absolute tolerance forwarded to ``numpy.testing.assert_allclose``.
    """
    assert len(logged_list) == len(expected_list), (
        f"Series count mismatch: logged {len(logged_list)}, "
        f"expected {len(expected_list)}.\n"
        f"Logged plt_funcs: {[s['plt_func'] for s in logged_list]}"
    )

    sorted_logged   = sorted(logged_list,   key=_series_sort_key)
    sorted_expected = sorted(expected_list, key=_series_sort_key)

    for logged, expected in zip(sorted_logged, sorted_expected):
        assert_series_matches(logged, expected, atol=atol)
        assert_lengths_consistent(logged)
