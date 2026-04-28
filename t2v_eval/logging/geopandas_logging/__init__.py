"""
GeoPandas monkey-patch registry.

Wires the single geopandas patch behind the public entry point
:func:`activate_geopandas_logging`.
"""

from .geopandas_geodataframe_plotting import activate_geodataframe_plot_dataframe


def activate_geopandas_logging() -> None:
    """Install all T2V monkey-patches on ``geopandas.plotting``."""
    activate_geodataframe_plot_dataframe()
