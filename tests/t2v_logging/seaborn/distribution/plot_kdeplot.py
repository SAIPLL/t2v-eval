"""
Test cases for ``sns.kdeplot`` logging.

How it is logged
----------------
KDE line  → ``ax.plot()``           → plt_func = "plot"
Filled KDE → ``ax.fill_between()`` → plt_func = "fill_between"

Strategy: partial check
-----------------------
The KDE grid and bandwidth depend on seaborn internals; coordinate values
are not verified.  Each case checks:
  - correct ``plt_func`` per series
  - correct series count (hue levels / fill)
  - length consistency (x and y same length within each series)

Empty-series artefact
---------------------
Same pattern as scatterplot: without hue, a n=0 series precedes the data.
With hue or fill, no artefact.
"""

import seaborn as sns

from tests.t2v_logging.seaborn.distribution._datasets import make_penguins


def case_basic(ax):
    """
    Basic KDE, no hue.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", kind="kde")

    → [empty "plot", KDE "plot"].
    """
    penguins = make_penguins()
    sns.kdeplot(data=penguins, x="flipper_length_mm", ax=ax)
    return [{"plt_func": "plot"}, {"plt_func": "plot"}]


def case_hue(ax):
    """
    KDE per species.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde")

    → 1 "plot" series per species (3 total), no artefact.
    """
    penguins = make_penguins()
    sns.kdeplot(data=penguins, x="flipper_length_mm", hue="species", ax=ax)
    n_hue = penguins["species"].nunique()
    return [{"plt_func": "plot"}] * n_hue


def case_filled(ax):
    """
    Filled KDE.
    Tutorial: sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True)

    → [empty "fill_between", filled "fill_between"].
    """
    penguins = make_penguins()
    sns.kdeplot(data=penguins, x="flipper_length_mm", fill=True, ax=ax)
    return [{"plt_func": "fill_between"}, {"plt_func": "fill_between"}]


def case_hue_filled(ax):
    """
    Filled KDE per species — no artefact.
    """
    penguins = make_penguins()
    sns.kdeplot(data=penguins, x="flipper_length_mm",
                hue="species", fill=True, ax=ax)
    n_hue = penguins["species"].nunique()
    return [{"plt_func": "fill_between"}] * n_hue
