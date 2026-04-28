"""
Logging tests for pandas pairwise / 2-D plot functions.

Plot types covered
------------------
- ``Series.plot`` / ``DataFrame.plot``  → ``plot_line.py``
- ``DataFrame.plot.scatter``            → ``plot_scatter.py``
- ``DataFrame.plot.area``               → ``plot_area.py``

How to add tests for a new function
------------------------------------
1. Create ``tests/t2v_logging/pandas/pairwise_data/plot_<func>.py``
   following the patterns in the existing files.
2. Add the module's dotted path to ``PLOT_MODULES`` below.
3. Run ``pytest tests/t2v_logging/test_pandas_pairwise.py -v``

Mapping to matplotlib
----------------------
Pandas delegates to matplotlib internally; the logged plt_func therefore
matches the underlying matplotlib call:

    pandas plot type       matplotlib call    logged plt_func
    ─────────────────────  ─────────────────  ───────────────
    Series/DataFrame.plot  ax.plot()          "plot"
    plot.scatter           ax.scatter()       "scatter"
    plot.area              ax.fill_between()  "fill_between"
                           ax.plot()          "plot"  (edge lines)
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
    "tests.t2v_logging.pandas.pairwise_data.plot_line",
    "tests.t2v_logging.pandas.pairwise_data.plot_scatter",
    "tests.t2v_logging.pandas.pairwise_data.plot_area",
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
    plt.close(fig)

    expected_series = resolve_expected(case_result)
    logged_series   = read_log_series(log_dir)

    assert len(logged_series) == len(expected_series), (
        f"Series count: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"Logged: {[s['plt_func'] for s in logged_series]}"
    )
    for logged, expected in zip(logged_series, expected_series):
        assert_series_matches(logged, expected)
        assert_lengths_consistent(logged)
