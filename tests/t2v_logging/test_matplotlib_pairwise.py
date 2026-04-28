"""
Logging tests for matplotlib 2-D (pairwise) plot functions.

How to add tests for a new function
------------------------------------
1. Create ``tests/t2v_logging/matplotlib/pairwise_data/plot_<func>.py``
   following the pattern in ``plot_scatter.py``:
   - Define one or more ``case_*(ax)`` functions.
   - Each function plots on *ax* and returns either:
       a. A list of expected-series dicts  (computed strategy), or
       b. A path to a golden log directory (golden strategy).

2. Add the module's dotted import path to ``PLOT_MODULES`` below.

3. Run ``pytest tests/t2v_logging/test_matplotlib_pairwise.py -v``

When to use each strategy
--------------------------
Computed  — ``scatter``, ``plot``: logged values match inputs directly.
Golden    — ``bar``, ``hist``, ``fill_between``: logged values are transformed
            (bar top-centres, polygon vertices, bin tops).  Run the script
            once, verify the JSON manually, save the dir as the golden reference.

Test assertions (per series)
-----------------------------
- **plt_func** — logged name matches expected.
- **series count** — number of logged series equals len(expected list).
- **length consistency** — all coordinate arrays (x, y, z, s, t) in the
  logged series have the same length.
- **value accuracy** — logged coordinates match expected values within
  ``atol=1e-4`` (skipped for coordinate keys absent from the expected dict).
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
# Register plot modules here — add a new line for each plot_<func>.py file
# ---------------------------------------------------------------------------
PLOT_MODULES = [
    "tests.t2v_logging.matplotlib.pairwise_data.plot_scatter",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_plot",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_bar",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_barh",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_fill_between",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_fill_betweenx",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_stackplot",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_stem",
    "tests.t2v_logging.matplotlib.pairwise_data.plot_stairs",
    # "tests.t2v_logging.matplotlib.pairwise_data.plot_axhline",
    # "tests.t2v_logging.matplotlib.pairwise_data.plot_axvline",
]


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------

def _collect_cases():
    """
    Import every module in PLOT_MODULES and collect its ``case_*`` functions.

    Returns a list of ``pytest.param(case_fn, id=<module>::<case_name>)``
    objects ready for ``@pytest.mark.parametrize``.
    """
    params = []
    for module_path in PLOT_MODULES:
        module = importlib.import_module(module_path)
        module_label = module_path.rsplit(".", 1)[-1]   # e.g. "plot_scatter"
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("case_"):
                params.append(
                    pytest.param(fn, id=f"{module_label}::{name}")
                )
    return params


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_fn", _collect_cases())
def test_logging(log_dir, case_fn):
    """
    Run *case_fn*, then verify the T2V log matches all expected series.

    Steps
    -----
    1. Call ``case_fn(ax)`` — plots on a fresh axes and returns expected.
    2. Resolve expected: list of dicts (computed) or path (golden dir).
    3. Read all series logged to *log_dir*.
    4. Assert series count, plt_func, length consistency, and value accuracy.
    """
    print(f"\n[log_dir]  {log_dir}")
    print(f"[case_fn]  {case_fn.__module__}.{case_fn.__name__}")

    fig, ax = plt.subplots()
    case_result = case_fn(ax)
    plt.close(fig)

    # Resolve: computed list OR golden log dir path
    expected_series = resolve_expected(case_result)
    logged_series   = read_log_series(log_dir)

    # --- series count -------------------------------------------------------
    assert len(logged_series) == len(expected_series), (
        f"Series count mismatch: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"Logged plt_funcs: {[s['plt_func'] for s in logged_series]}"
    )

    # --- per-series checks --------------------------------------------------
    for logged, expected in zip(logged_series, expected_series):

        # plt_func + value accuracy (only for keys present in expected)
        assert_series_matches(logged, expected)

        # length consistency (x/y/z/s/t all same length within each series)
        assert_lengths_consistent(logged)
