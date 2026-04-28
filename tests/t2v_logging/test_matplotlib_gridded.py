"""
Logging tests for matplotlib gridded / matrix plot functions.

How to add tests for a new function
------------------------------------
1. Create ``tests/t2v_logging/matplotlib/gridded/plot_<func>.py``
   following the pattern in ``plot_imshow.py``.

2. Add the module's dotted import path to ``PLOT_MODULES`` below.

3. Run ``pytest tests/t2v_logging/test_matplotlib_gridded.py -v``
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

PLOT_MODULES = [
    "tests.t2v_logging.matplotlib.gridded.plot_imshow",
    "tests.t2v_logging.matplotlib.gridded.plot_pcolormesh",
    "tests.t2v_logging.matplotlib.gridded.plot_contour",
]


def _collect_cases():
    params = []
    for module_path in PLOT_MODULES:
        module = importlib.import_module(module_path)
        module_label = module_path.rsplit(".", 1)[-1]
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("case_"):
                params.append(pytest.param(fn, id=f"{module_label}::{name}"))
    return params


@pytest.mark.parametrize("case_fn", _collect_cases())
def test_logging(log_dir, case_fn):
    print(f"\n[log_dir]  {log_dir}")
    print(f"[case_fn]  {case_fn.__module__}.{case_fn.__name__}")

    fig, ax = plt.subplots()
    case_result = case_fn(ax)
    plt.close(fig)

    expected_series = resolve_expected(case_result)
    logged_series   = read_log_series(log_dir)

    assert len(logged_series) == len(expected_series), (
        f"Series count mismatch: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"Logged plt_funcs: {[s['plt_func'] for s in logged_series]}"
    )

    for logged, expected in zip(logged_series, expected_series):
        assert_series_matches(logged, expected)
        assert_lengths_consistent(logged)
