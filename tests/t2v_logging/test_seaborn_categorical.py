"""
Logging tests for seaborn categorical plot functions.

Plot types covered
------------------
- ``sns.stripplot`` / ``sns.swarmplot``  → ``plot_strip_swarm.py``
- ``sns.boxplot``  / ``sns.violinplot``  → ``plot_box_violin.py``
- ``sns.barplot``  / ``sns.countplot`` / ``sns.pointplot`` → ``plot_estimate.py``
- ``sns.catplot`` (figure-level, all kinds)

Logged plt_func mapping
------------------------
    seaborn function      matplotlib call   plt_func
    ───────────────────   ───────────────   ────────────
    stripplot/swarmplot   ax.scatter()      "scatter"
    boxplot/violinplot    ax.bxp()          "bxp"
    barplot               ax.bar()          "bar"
                          ax.plot()         "plot"  (CI lines)
    countplot             ax.bar()          "bar"
    pointplot             ax.plot()         "plot"

Key patterns
------------
- **Figure-level** (catplot): no empty legend artefact.
- **Axes-level**: 1 extra empty-series artefact (n=0 scatter / fill_between / bar).
- **UnorderedSeries**: used when seaborn drawing order is category-controlled.
- **Golden-path** (``None``): used for complex hue × category combinations.
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


_VALID_CAT_FUNCS = {"scatter", "bxp", "bar", "barh", "plot", "fill_between"}

PLOT_MODULES = [
    "tests.t2v_logging.seaborn.categorical.plot_strip_swarm",
    "tests.t2v_logging.seaborn.categorical.plot_box_violin",
    "tests.t2v_logging.seaborn.categorical.plot_estimate",
    "tests.t2v_logging.seaborn.categorical.plot_catplot",
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
            assert s["plt_func"] in _VALID_CAT_FUNCS, \
                f"Unexpected plt_func {s['plt_func']!r}"
            assert_lengths_consistent(s)
        return

    expected_series = resolve_expected(case_result)

    # UnorderedSeries: order-independent matching
    if isinstance(expected_series, UnorderedSeries):
        assert_series_set_matches(logged_series, expected_series)
        return

    # Ordered matching
    assert len(logged_series) == len(expected_series), (
        f"Series count: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"Logged: {[s['plt_func'] for s in logged_series]}"
    )
    for logged, expected in zip(logged_series, expected_series):
        assert_series_matches(logged, expected)
        assert_lengths_consistent(logged)
