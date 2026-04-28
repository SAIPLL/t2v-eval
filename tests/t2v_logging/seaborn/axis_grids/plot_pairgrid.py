"""
Test cases for ``sns.PairGrid``, ``sns.pairplot``.

Key observation: PairGrid has artefacts
-----------------------------------------
Unlike ``FacetGrid.map()``, ``PairGrid.map()`` calls through the axes-level
code path, producing the usual empty-series artefact per cell.

Series count formulas
---------------------
PairGrid.map(scatterplot)          → n_vars² cells × 2 (artefact+data) scatter
PairGrid.map_diag(histplot)        → n_vars cells × 2 (artefact+data) bar
    + map_offdiag(scatterplot)     → (n_vars²-n_vars) cells × 2 scatter
pairplot(hue=species)              → n_vars diagonal × n_species fill_between
                                   + (n_vars²-n_vars) off-diag × 1 scatter

pairplot diagonal
-----------------
Default diagonal is ``histplot`` (no hue) → 1 bar per diagonal cell.
With ``hue=`` → KDE fill per species → n_species fill_between per cell.

All cases use partial checks (count + plt_func).
"""

import seaborn as sns

from tests.t2v_logging.seaborn.axis_grids._datasets import make_iris, make_tips


def case_pairgrid_map_scatter(ax):
    """
    Tutorial: g = PairGrid(iris); g.map(sns.scatterplot)
    4 vars × 4 vars = 16 cells × 2 (artefact + data) = 32 "scatter".
    """
    iris   = make_iris()
    n_vars = 4  # sepal_length, sepal_width, petal_length, petal_width
    sns.PairGrid(iris).map(sns.scatterplot)
    return [{"plt_func": "scatter"}] * (n_vars ** 2 * 2)


def case_pairgrid_diag_hist_offdiag_scatter(ax):
    """
    Tutorial: g.map_diag(histplot); g.map_offdiag(scatterplot)
    Diagonal (4 cells) × 2 = 8 "bar".
    Off-diagonal (12 cells) × 2 = 24 "scatter".
    Total = 32 series.
    Golden-path: logged order depends on axes-object IDs (non-deterministic
    across test runs when other tests precede this one).
    """
    iris = make_iris()
    g    = sns.PairGrid(iris)
    g.map_diag(sns.histplot)
    g.map_offdiag(sns.scatterplot)
    return None   # golden-path


def case_pairgrid_upper_lower_diag(ax):
    """
    Tutorial: g.map_upper(scatterplot); g.map_lower(kdeplot); g.map_diag(kdeplot)
    Upper triangle (6 cells) × 2 scatter = 12 scatter.
    Diagonal (4 cells) × 1 KDE plot = 4 plot.
    Lower triangle (6 cells) × 1 KDE plot = 6 plot.
    Total = 12 scatter + 10 plot = 22 series.
    (kdeplot has no artefact via FacetGrid-style call; scatter has artefact.)
    """
    iris   = make_iris()
    n_vars = 4
    n_upper   = n_vars * (n_vars - 1) // 2
    n_lower   = n_upper
    g = sns.PairGrid(iris)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=3, legend=False)
    return None   # golden-path: complex mix of scatter+plot series


def case_pairgrid_regplot(ax):
    """
    Tutorial: g = PairGrid(tips, y_vars=["tip"], x_vars=["total_bill","size"])
              g.map(sns.regplot, color=".3")
    1 row × 2 cols = 2 cells.
    Each: regplot via PairGrid → scatter + line + CI (NO artefact — PairGrid
    suppresses the empty-series legend handle that axes-level regplot creates).
    Total = 2 × 3 = 6 series.
    """
    tips = make_tips()
    g    = sns.PairGrid(tips, y_vars=["tip"],
                        x_vars=["total_bill", "size"], height=4)
    g.map(sns.regplot, color=".3")
    n_cells = 2   # 1 y_var × 2 x_vars
    # Each cell: scatter + regression line + CI (no artefact via PairGrid)
    return ([{"plt_func": "scatter"}] +
            [{"plt_func": "plot"}] +
            [{"plt_func": "fill_between"}]) * n_cells


def case_pairplot_hue(ax):
    """
    Tutorial: sns.pairplot(iris, hue="species", height=2.5)
    4×4 grid — diagonal: 3 fill_between/species × 4 = 12; off-diag: 12 scatter.
    Total = 24 series. Golden-path (axes-ID ordering non-deterministic).
    """
    iris = make_iris()
    sns.pairplot(iris, hue="species", height=2.5)
    return None   # golden-path


def case_pairplot_no_hue(ax):
    """
    Tutorial: sns.pairplot(iris)  (no hue)
    Diagonal: histplot 2 bar/cell × 4 = 8; off-diagonal: 2 scatter/cell × 12 = 24.
    Total = 32 series. Golden-path (axes-ID ordering non-deterministic).
    """
    iris = make_iris()
    sns.pairplot(iris, height=2.5)
    return None   # golden-path
