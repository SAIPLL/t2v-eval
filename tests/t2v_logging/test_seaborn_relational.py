"""
Logging tests for seaborn "Visualizing statistical relationships" plots.

Covers
------
- ``sns.scatterplot`` / ``sns.relplot(kind="scatter")``  → ``plot_scatterplot.py``
- ``sns.lineplot``    / ``sns.relplot(kind="line")``     → ``plot_lineplot.py``

Mapping to matplotlib plt_func
--------------------------------
    seaborn function          matplotlib call      logged plt_func
    ──────────────────────    ─────────────────    ───────────────
    scatterplot               ax.scatter()         "scatter"
    lineplot                  ax.plot()            "plot"
    lineplot (with CI)        ax.fill_between()    "fill_between"

Add new tests
-------------
1. Create ``tests/t2v_logging/seaborn/relational/plot_<func>.py`` with
   ``case_*`` functions.
2. Add the module path to ``PLOT_MODULES`` below.
3. Run ``pytest tests/t2v_logging/test_seaborn_relational.py -v``
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
    "tests.t2v_logging.seaborn.relational.plot_scatterplot",
    "tests.t2v_logging.seaborn.relational.plot_lineplot",
    "tests.t2v_logging.seaborn.relational.plot_relplot",
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
    plt.close("all")   # closes both the test figure and any relplot FacetGrid

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
