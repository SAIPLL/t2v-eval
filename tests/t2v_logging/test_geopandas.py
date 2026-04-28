"""
Logging tests for geopandas world heatmap.

Covers
------
- ``GeoDataFrame.plot(column=...)``  →  ``plot_world_heatmap.py``

How geopandas is logged
------------------------
``activate_geodataframe_plot_dataframe`` patches
``geopandas.plotting.plot_dataframe``.  Each polygon part in the rendered
``PatchCollection`` becomes one ``"geo"`` series via
``patchcollection_to_logdata``::

    plt_func = "geo"
    x        = polygon border x-coords (longitude)
    y        = polygon border y-coords (latitude)
    z        = scalar value (heatmap colour variable, repeated per vertex)
    color    = RGBA tuple

One series per polygon part means MultiPolygon countries (islands, exclaves)
produce MORE series than the GeoDataFrame row count.

Valid logged funcs
------------------
Only ``"geo"`` is expected.  A colourbar/legend does not fire the patch.
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
    "tests.t2v_logging.geopandas.plot_world_heatmap",
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

    fig, ax = plt.subplots(figsize=(15, 8))
    case_result = case_fn(ax)
    plt.close("all")

    logged_series = read_log_series(log_dir)
    expected_series = resolve_expected(case_result)

    assert len(logged_series) == len(expected_series), (
        f"Series count: logged {len(logged_series)}, "
        f"expected {len(expected_series)}.\n"
        f"All logged plt_funcs: "
        f"{set(s['plt_func'] for s in logged_series)}"
    )
    for logged, expected in zip(logged_series, expected_series):
        assert_series_matches(logged, expected)
        assert_lengths_consistent(logged)
