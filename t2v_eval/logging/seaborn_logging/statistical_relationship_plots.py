"""
Monkey-patches for seaborn statistical-relationship plots.

Unlike the matplotlib patches, this file replaces the *entire* internal method
``seaborn.relational._ScatterPlotter.plot`` rather than wrapping it.  This is
necessary because the method calls ``ax.scatter()`` internally — a thin wrapper
would trigger the matplotlib scatter patch and double-log the data.  Logging is
therefore disabled for the duration of the seaborn computation and re-enabled
immediately before the T2V log call at the end.

Call :func:`activate_sns_scatterplot` to install the patch.
"""

import matplotlib as mpl
import numpy as np
import seaborn.relational
from seaborn.relational import _get_transform_functions, _scatter_legend_artist
from seaborn.utils import adjust_legend_subtitles, normalize_kwargs

from t2v_eval.logging.utils import get_variable, log_data, pathcollection_to_logdata, set_variable


def activate_sns_scatterplot() -> None:
    """Replace ``_ScatterPlotter.plot`` with a logging-aware implementation."""

    def plot(self, ax, kws):
        # Disable logging for the duration of this call so the internal
        # ax.scatter() does not trigger the matplotlib scatter patch.
        was_logging = get_variable("T2V_ISLOG")
        set_variable("T2V_ISLOG", False)

        # --- Seaborn scatter plot implementation ----------------------------

        data = self.comp_data.dropna()
        if data.empty:
            return

        kws = normalize_kwargs(kws, mpl.collections.PathCollection)

        empty = np.full(len(data), np.nan)
        x = data.get("x", empty)
        y = data.get("y", empty)

        _, inv_x = _get_transform_functions(ax, "x")
        _, inv_y = _get_transform_functions(ax, "y")
        x, y = inv_x(x), inv_y(y)

        if "style" in self.variables:
            example_level  = self._style_map.levels[0]
            example_marker = self._style_map(example_level, "marker")
            kws.setdefault("marker", example_marker)

        m = kws.get("marker", mpl.rcParams.get("marker", "o"))
        if not isinstance(m, mpl.markers.MarkerStyle):
            m = mpl.markers.MarkerStyle(m)
        if m.is_filled():
            kws.setdefault("edgecolor", "w")

        points = ax.scatter(x=x, y=y, **kws)

        if "hue" in self.variables:
            points.set_facecolors(self._hue_map(data["hue"]))
        if "size" in self.variables:
            points.set_sizes(self._size_map(data["size"]))
        if "style" in self.variables:
            points.set_paths([self._style_map(val, "path") for val in data["style"]])

        if "linewidth" not in kws:
            sizes = points.get_sizes()
            linewidth = 0.08 * np.sqrt(np.percentile(sizes, 10))
            points.set_linewidths(linewidth)
            kws["linewidth"] = linewidth

        self._add_axis_labels(ax)
        if self.legend:
            attrs = {"hue": "color", "size": "s", "style": None}
            self.add_legend_data(ax, _scatter_legend_artist, kws, attrs)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                adjust_legend_subtitles(ax.legend(title=self.legend_title))

        # --- T2V logging ---------------------------------------------------

        set_variable("T2V_ISLOG", was_logging)
        if was_logging is True:
            ax.figure.canvas.draw()
            log_data(ax, "scatter", [], {}, pathcollection_to_logdata(points))

    seaborn.relational._ScatterPlotter.plot = plot
