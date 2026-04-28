"""
Logging tests for seaborn multi-subplot grid functions.

Covers
------
- ``sns.FacetGrid`` → ``plot_facetgrid.py``
- ``sns.PairGrid`` / ``sns.pairplot`` → ``plot_pairgrid.py``

Series count rules
------------------
**FacetGrid** (no artefacts):
    n_series = n_col × n_row × n_hue × n_per_mapped_fn

**PairGrid** (artefacts per cell):
    n_series = n_cells × n_per_cell × n_artefact_multiplier

The expected lists contain only plt_func checks (no coordinate values)
because the logged data depends on layout geometry and seaborn internals.
"""

import importlib
import inspect

import matplotlib.pyplot as plt
import pytest

from tests.t2v_logging.helpers import (
    assert_lengths_consistent,
    assert_series_matches,
    read_log_series,
    resolve_expected,
)

_VALID_GRID_FUNCS = {
    "scatter", "bar", "barh", "plot", "fill_between",
    "hexbin", "bxp", "hist2d", "contour", "contourf",
}

PLOT_MODULES = [
    "tests.t2v_logging.seaborn.axis_grids.plot_facetgrid",
    "tests.t2v_logging.seaborn.axis_grids.plot_pairgrid",
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

    # Golden-path: structural check only
    if case_result is None:
        assert len(logged_series) > 0, "No series logged."
        for s in logged_series:
            assert s["plt_func"] in _VALID_GRID_FUNCS, \
                f"Unexpected plt_func {s['plt_func']!r}"
            assert_lengths_consistent(s)
        return

    expected_series = resolve_expected(case_result)

    assert len(logged_series) == len(expected_series), (
        f"Series count: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"Logged plt_funcs: {[s['plt_func'] for s in logged_series]}"
    )
    for logged, expected in zip(logged_series, expected_series):
        assert_series_matches(logged, expected)
        assert_lengths_consistent(logged)
