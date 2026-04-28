"""
Logging tests for matplotlib statistical-distribution plot functions.

How to add tests for a new function
------------------------------------
1. Create ``tests/t2v_logging/matplotlib/statistical_distributions/plot_<func>.py``
   following the pattern in ``plot_hist.py``:
   - Define one or more ``case_*(ax)`` functions.
   - Each function plots on *ax* and returns a list of expected-series dicts
     (or a path to a golden log directory).

2. Add the module's dotted import path to ``PLOT_MODULES`` below.

3. Run ``pytest tests/t2v_logging/test_matplotlib_distribution.py -v``

Test assertions (per series)
-----------------------------
- **plt_func** — logged name matches expected.
- **series count** — number of logged series equals len(expected list).
- **length consistency** — all coordinate arrays (x, y, z, s, t) in the
  logged series have the same length.
- **value accuracy** — logged coordinates match expected values within
  ``atol=1e-4`` (skipped for keys absent from the expected dict).
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

# ---------------------------------------------------------------------------
# Register plot modules here
# ---------------------------------------------------------------------------
PLOT_MODULES = [
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_hist",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_boxplot",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_violinplot",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_errorbar",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_eventplot",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_pie",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_hexbin",
    "tests.t2v_logging.matplotlib.statistical_distributions.plot_hist2d",
]


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _collect_cases():
    """Collect all ``case_*`` functions from registered plot modules."""
    params = []
    for module_path in PLOT_MODULES:
        module = importlib.import_module(module_path)
        module_label = module_path.rsplit(".", 1)[-1]
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("case_"):
                params.append(pytest.param(fn, id=f"{module_label}::{name}"))
    return params


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_fn", _collect_cases())
def test_logging(log_dir, case_fn):
    """
    Run *case_fn*, then verify the T2V log matches all expected series.
    """
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
