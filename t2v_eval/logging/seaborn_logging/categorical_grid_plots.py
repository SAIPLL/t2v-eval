"""
Monkey-patches for seaborn categorical grid / heatmap plots.

Both patches suppress the colour-bar (``cbar``) when T2V logging is active so
that the colour-bar does not produce an extra axes that interferes with the
logged data.

Call :func:`activate_sns_categorical_grid_plots` to install all patches.
"""

import seaborn
import seaborn.matrix

from t2v_eval.logging.utils import get_variable


# ---------------------------------------------------------------------------
# Per-method activations
# ---------------------------------------------------------------------------

def activate_sns_matrix_heatmapper_plot() -> None:
    """Patch ``_HeatMapper.plot`` to disable the colour-bar during logging."""
    _orig = seaborn.matrix._HeatMapper.plot

    def plot(self, ax, cax, kws):
        if get_variable("T2V_ISLOG"):
            self.cbar = False
        return _orig(self, ax, cax, kws)

    seaborn.matrix._HeatMapper.plot = plot


def activate_sns_heatmap() -> None:
    """Patch ``seaborn.heatmap`` to disable the colour-bar during logging."""
    _orig = seaborn.heatmap

    def heatmap(*args, **kwargs):
        if get_variable("T2V_ISLOG"):
            kwargs["cbar"] = False
        return _orig(*args, **kwargs)

    seaborn.heatmap        = heatmap
    seaborn.matrix.heatmap = heatmap


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def activate_sns_categorical_grid_plots() -> None:
    """Install all categorical-grid monkey-patches on ``seaborn``."""
    activate_sns_matrix_heatmapper_plot()
    activate_sns_heatmap()
