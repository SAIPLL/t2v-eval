"""
Logging tests for pandas statistical-distribution plot functions.

Plot types covered
------------------
- ``Series/DataFrame.plot.bar``    → ``plot_bar.py``
- ``Series.plot.barh``             → ``plot_barh.py``
- ``Series/DataFrame.plot.hist``   → ``plot_hist.py``
- ``DataFrame.plot.box``           → ``plot_box.py``
- ``Series.plot.pie``              → ``plot_pie.py``
- ``Series/DataFrame.plot.kde``    → ``plot_kde.py``
- ``DataFrame.plot.hexbin``        → ``plot_hexbin.py``

Mapping to matplotlib
----------------------
    pandas plot type   matplotlib call       logged plt_func
    ─────────────────  ───────────────────   ───────────────
    plot.bar           ax.bar()              "bar"
    plot.barh          ax.barh()             "barh"
    plot.hist          ax.hist()             "hist"
    plot.box           ax.bxp()              "bxp"
    plot.pie           ax.pie()              "pie"
    plot.kde           ax.plot() (KDE curve) "plot"
    plot.hexbin        ax.hexbin()           "hexbin"
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
    "tests.t2v_logging.pandas.statistical_distributions.plot_bar",
    "tests.t2v_logging.pandas.statistical_distributions.plot_barh",
    "tests.t2v_logging.pandas.statistical_distributions.plot_hist",
    "tests.t2v_logging.pandas.statistical_distributions.plot_box",
    "tests.t2v_logging.pandas.statistical_distributions.plot_pie",
    "tests.t2v_logging.pandas.statistical_distributions.plot_kde",
    "tests.t2v_logging.pandas.statistical_distributions.plot_hexbin",
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
