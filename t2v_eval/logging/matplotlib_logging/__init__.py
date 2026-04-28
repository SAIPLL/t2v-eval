"""
Matplotlib monkey-patch registry.

Imports one ``activate_*`` function from each submodule and wires them
together behind the single public entry point
:func:`activate_matplotlib_logging`.
"""

from .matplotlib_3D import activate_matplotlib_3D_data_logging
from .matplotlib_distribution_data import activate_matplotlib_distribution_data_logging
from .matplotlib_grided_data import activate_matplotlib_gridded_data_logging
from .matplotlib_layout import activate_matplotlib_layout_logging
from .matplotlib_pairwise_data import activate_matplotlib_pairwise_data_logging


def activate_matplotlib_logging() -> None:
    """Install all T2V monkey-patches on ``matplotlib`` and ``mpl_toolkits``."""
    activate_matplotlib_pairwise_data_logging()
    activate_matplotlib_distribution_data_logging()
    activate_matplotlib_gridded_data_logging()
    activate_matplotlib_3D_data_logging()
    activate_matplotlib_layout_logging()
