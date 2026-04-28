"""
Logging tests for seaborn distribution plot functions.

Plot types covered
------------------
- ``sns.histplot``   → ``plot_histplot.py``   (plt_func = "bar")
- ``sns.kdeplot``    → ``plot_kdeplot.py``    (plt_func = "plot" / "fill_between")
- ``sns.ecdfplot``   → ``plot_ecdfplot.py``   (plt_func = "plot")
- ``sns.displot``    → ``plot_displot.py``    (figure-level, all kinds)
- ``sns.jointplot``  → ``plot_jointplot.py``  (figure-level, scatter + kde)

Golden-path cases
-----------------
Case functions that return ``None`` are "golden-path" tests: the series
count and plt_func of every logged series are verified (all must be one of
the expected distribution plt_funcs), but coordinate values are not checked.
This covers complex cases like 2-D KDE contours where the exact segment
count is data-dependent.
"""

import importlib
import inspect

import matplotlib.pyplot as plt
import pytest

from tests.t2v_logging.helpers import (
    UnorderedSeries,
    assert_lengths_consistent,
    assert_series_matches,
    assert_series_set_matches,
    read_log_series,
    resolve_expected,
)

PLOT_MODULES = [
    "tests.t2v_logging.seaborn.distribution.plot_histplot",
    "tests.t2v_logging.seaborn.distribution.plot_kdeplot",
    "tests.t2v_logging.seaborn.distribution.plot_ecdfplot",
    "tests.t2v_logging.seaborn.distribution.plot_displot",
    "tests.t2v_logging.seaborn.distribution.plot_jointplot",
]

_VALID_DIST_FUNCS = {"bar", "barh", "plot", "fill_between",
                     "scatter", "pcolormesh", "contour", "contourf"}


def _collect_cases():
    params = []
    for module_path in PLOT_MODULES:
        module = importlib.import_module(module_path)
        label  = module_path.rsplit(".", 1)[-1]
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("case_"):
                params.append(pytest.param(fn, id=f"{label}::{name}"))
    return params


@pytest.mark.parametrize("case_fn", _collect_cases())
def test_logging(log_dir, case_fn):
    print(f"\n[log_dir]  {log_dir}")
    print(f"[case_fn]  {case_fn.__module__}.{case_fn.__name__}")

    fig, ax = plt.subplots()
    case_result = case_fn(ax)
    plt.close("all")

    logged_series = read_log_series(log_dir)

    if case_result is None:
        # Golden-path: verify series exist, plt_funcs are valid distribution
        # types, and all coordinate arrays are length-consistent.
        assert len(logged_series) > 0, "No series logged."
        for s in logged_series:
            assert s["plt_func"] in _VALID_DIST_FUNCS, \
                f"Unexpected plt_func {s['plt_func']!r}"
            assert_lengths_consistent(s)
        return

    expected_series = resolve_expected(case_result)

    if isinstance(expected_series, UnorderedSeries):
        # Hue / facet order is seaborn-controlled — match by median value
        assert_series_set_matches(logged_series, expected_series)
        return

    assert len(logged_series) == len(expected_series), (
        f"Series count: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"Logged: {[s['plt_func'] for s in logged_series]}"
    )
    for logged, expected in zip(logged_series, expected_series):
        assert_series_matches(logged, expected)
        assert_lengths_consistent(logged)
