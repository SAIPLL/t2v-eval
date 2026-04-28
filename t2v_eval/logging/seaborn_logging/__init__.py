"""
Seaborn monkey-patch registry.

Imports one ``activate_*`` function from each submodule and wires them
together behind the single public entry point
:func:`activate_seaborn_logging`.
"""

from .categorical_distribution_plots import activate_sns_categorical_distribution_plots
from .categorical_estimate_plots import activate_sns_categorical_estimate_plots
from .categorical_grid_plots import activate_sns_categorical_grid_plots
from .categorical_scatter_plots import activate_sns_categorical_scatter_plots
from .statistical_relationship_plots import activate_sns_scatterplot


def activate_seaborn_logging() -> None:
    """Install all T2V monkey-patches on ``seaborn`` and ``seaborn.categorical``."""
    activate_sns_categorical_distribution_plots()
    activate_sns_categorical_estimate_plots()
    activate_sns_categorical_grid_plots()
    activate_sns_categorical_scatter_plots()
    activate_sns_scatterplot()
