"""
Logging tests for matplotlib 3-D plot functions.

Why a separate runner?
-----------------------
3-D plots require ``Axes3D`` (``projection='3d'``), not the default
``Axes``.  This runner creates a 3-D axes for every test, while the other
runners use standard 2-D axes.

How to add tests for a new 3-D function
-----------------------------------------
1. Create ``tests/t2v_logging/matplotlib/threed/plot_<func>.py``
   following the pattern in ``plot_scatter3d.py``.

2. Add the module's dotted import path to ``PLOT_MODULES`` below.

3. Run ``pytest tests/t2v_logging/test_matplotlib_3d.py -v``
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
    "tests.t2v_logging.matplotlib.threed.plot_scatter3d",
    "tests.t2v_logging.matplotlib.threed.plot_bar3d",
    "tests.t2v_logging.matplotlib.threed.plot_surface3d",
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
    """
    Run *case_fn* on a 3-D axes, then verify the T2V log matches all
    expected series.
    """
    print(f"\n[log_dir]  {log_dir}")
    print(f"[case_fn]  {case_fn.__module__}.{case_fn.__name__}")

    fig = plt.figure()
    ax  = fig.add_subplot(projection="3d")
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
