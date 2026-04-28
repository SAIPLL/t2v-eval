"""
Logging tests for seaborn regression plot functions.

Plot types covered
------------------
- ``sns.regplot``   (axes-level) → ``plot_regplot.py``
- ``sns.residplot`` (axes-level) → ``plot_residplot.py``
- ``sns.lmplot``    (figure-level) → ``plot_lmplot.py``

Logged plt_func mapping
------------------------
    Function            matplotlib call       plt_func
    ──────────────────  ──────────────────    ──────────────
    regplot / lmplot    ax.scatter()          "scatter"
    regplot / lmplot    ax.plot()             "plot"
    regplot / lmplot    ax.fill_between()     "fill_between"
    residplot           ax.axhline()          "axhline"
    residplot           ax.scatter()          "scatter"

Key patterns
------------
- **Axes-level** (regplot, residplot): 1 empty "plot" artefact (n=0).
- **Figure-level** (lmplot): no artefact.
- **Scatter** x/y values match input data → checked precisely.
- **Regression line** (n=100) and **CI band** (n=203) → partial checks.
- **UnorderedSeries**: used for hue cases where draw order is unspecified.
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

_VALID_REG_FUNCS = {"scatter", "plot", "fill_between", "axhline"}

PLOT_MODULES = [
    "tests.t2v_logging.seaborn.regression.plot_regplot",
    "tests.t2v_logging.seaborn.regression.plot_residplot",
    "tests.t2v_logging.seaborn.regression.plot_lmplot",
]


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
        assert len(logged_series) > 0, "No series logged."
        for s in logged_series:
            assert s["plt_func"] in _VALID_REG_FUNCS, \
                f"Unexpected plt_func {s['plt_func']!r}"
            assert_lengths_consistent(s)
        return

    expected_series = resolve_expected(case_result)

    if isinstance(expected_series, UnorderedSeries):
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
