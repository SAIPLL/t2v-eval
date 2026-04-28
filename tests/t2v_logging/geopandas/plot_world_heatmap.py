"""
Test cases for geopandas world heatmap logging.

Dataset
-------
``countries.geojson`` — 180 countries with ISO codes and polygon geometries.
A synthetic scalar column ("value") is added as the heatmap variable.

How it is logged
----------------
``activate_geodataframe_plot_dataframe`` patches
``geopandas.plotting.plot_dataframe``.  When ``gdf.plot(column=...)`` is
called, it logs **one "geo" series per polygon path** via
``patchcollection_to_logdata``.

Because many countries are MultiPolygon (islands, exclaves, etc.), the total
series count EXCEEDS the row count of the GeoDataFrame:

    n_series = n_total_polygon_parts  (= 292 for this GeoJSON)
    not simply n_countries            (= 180)

Logged structure per series
----------------------------
    plt_func = "geo"
    x        = polygon border x-coordinates (longitude)
    y        = polygon border y-coordinates (latitude)
    z        = scalar value mapped to colour (one value, repeated per vertex)
    color    = RGBA tuple from the colourmap

Strategy: computed
------------------
- ``plt_func`` = "geo" for every series — verified.
- ``n_series`` = 292 — exact count verified (fixed by GeoJSON polygon count).
- Length consistency (x, y, z same length per series) — verified.
- Coordinate values are not checked (polygon geometry is complex, taken from
  the GeoJSON itself).
"""

from pathlib import Path

import geopandas as gpd
import numpy as np

_GEOJSON = Path(__file__).parent / "countries.geojson"
# Total polygon parts in countries.geojson (180 countries, some MultiPolygon)
_N_POLYGON_PARTS = 292


def _load_world(seed: int = 42) -> gpd.GeoDataFrame:
    """Load the GeoJSON and add a synthetic heatmap column."""
    gdf = gpd.read_file(_GEOJSON)
    rng = np.random.default_rng(seed)
    gdf["value"] = rng.uniform(0, 100, len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def case_world_heatmap_basic(ax):
    """
    Basic world heatmap — one colour value per country.

    gdf.plot(column="value", ax=ax, cmap="YlOrRd")

    → 292 "geo" series (one per polygon part, including MultiPolygon pieces).
    """
    gdf = _load_world()
    gdf.plot(column="value", ax=ax, cmap="YlOrRd")
    return [{"plt_func": "geo"}] * _N_POLYGON_PARTS


def case_world_heatmap_with_legend(ax):
    """
    Heatmap with a colourbar legend.
    The legend is drawn in a separate axes (colourbar) which does NOT fire
    the ``plot_dataframe`` patch → same n_series as the basic case.
    """
    gdf = _load_world()
    gdf.plot(column="value", ax=ax, cmap="Blues", legend=True)
    return [{"plt_func": "geo"}] * _N_POLYGON_PARTS


def case_world_heatmap_missing_data(ax):
    """
    Heatmap with some countries having NaN — geopandas renders missing
    countries in the default "missing" colour (still logs all polygon parts).
    """
    gdf = _load_world()
    # Set first 10 countries to NaN
    gdf.loc[:9, "value"] = np.nan
    gdf.plot(column="value", ax=ax, cmap="RdYlGn",
             missing_kwds={"color": "lightgrey"})
    return [{"plt_func": "geo"}] * _N_POLYGON_PARTS


def case_world_heatmap_subsets(ax):
    """
    Heatmap of a subset (e.g., only countries with ISO3 starting with 'A').
    Only the matching polygon parts are logged.
    """
    gdf   = _load_world()
    sub   = gdf[gdf["ISO3166-1-Alpha-3"].str.startswith("A", na=False)].copy()
    # Count polygon parts in the subset
    n_parts = sum(
        1 if geom.geom_type == "Polygon" else len(geom.geoms)
        for geom in sub.geometry
    )
    sub.plot(column="value", ax=ax, cmap="viridis")
    return [{"plt_func": "geo"}] * n_parts
