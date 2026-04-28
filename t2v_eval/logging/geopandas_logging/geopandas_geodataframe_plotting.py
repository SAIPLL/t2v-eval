"""
Monkey-patch for ``geopandas.plotting.plot_dataframe``.

Intercepts geo-dataframe plot calls to capture the ``PatchCollection`` objects
added to the axes, converting them to T2V log-data format.

Call :func:`activate_geodataframe_plot_dataframe` to install the patch.
"""

import geopandas.plotting

from t2v_eval.logging.utils import log_data, logging_paused, patchcollection_to_logdata


def activate_geodataframe_plot_dataframe() -> None:
    """Patch ``geopandas.plotting.plot_dataframe`` to log geo series data."""
    _orig = geopandas.plotting.plot_dataframe

    def plot_dataframe(self, *args, **kwargs):
        # Snapshot collections already on the axes so we can identify the new
        # ones added by this call.
        existing_ax = kwargs.get("ax")
        pre_collections = set(existing_ax.collections) if existing_ax is not None else set()

        with logging_paused() as was_logging:
            ax = _orig(self, *args, **kwargs)

        if was_logging:
            ax.figure.canvas.draw()
            data_series = [
                series
                for collection in ax.collections
                if collection not in pre_collections
                for series in patchcollection_to_logdata(collection)
            ]
            log_data(ax, "geo", [],
                     {"x_lims": ax.get_xlim(), "y_lims": ax.get_ylim()},
                     data_series=data_series)

        return ax

    geopandas.plotting.plot_dataframe = plot_dataframe
