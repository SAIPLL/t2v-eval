"""
Re-plot utility for T2V log directories.

Reads the JSON files produced by ``activate_t2v_logging()`` and renders each
logged series as a scatter overlay (red dots) on a reconstructed figure, then
saves a ``merge_images_<fig_id>.png`` file alongside the logs.

Entry point: :func:`replot_figure`
"""

import glob
import json
import math
import os
import traceback
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from t2v_eval.evaluation.metric_utils import filter_out_invalid_values
from t2v_eval.logging import set_variable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESERVED_PREFIXES = ("variables", "evaluation", "execution")

# Keyword args shared by all red-dot overlay scatter calls
_OVERLAY = dict(color="red", s=5, zorder=math.inf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_axes_func(ax, func_name: str):
    """
    Return ``ax.<func_name>`` if it exists, otherwise ``None``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    func_name : str
        Name of the method to look up on *ax*.
    """
    return getattr(ax, func_name, None)


def convert_to_masked_array(data: list) -> np.ma.MaskedArray:
    """
    Convert a nested list to a NumPy masked array, treating ``None`` as NaN.

    Parameters
    ----------
    data : list of lists
        2-D data where individual values may be ``None``.
    """
    arr = np.array(
        [[np.nan if v is None else v for v in row] for row in data],
        dtype=float,
    )
    return np.ma.masked_invalid(arr)


def _try_original(func, args: list, kargs: dict) -> None:
    """Call ``func(*args, **kargs)``, silently ignoring any exception."""
    try:
        func(*args, **kargs)
    except Exception:
        pass


def _ensure_3d(ax, fig, mpl_axes: list, idx: int):
    """
    Upgrade *ax* to a 3-D projection if it is not already one.

    Updates ``mpl_axes[idx]`` in-place so subsequent layout calls use the
    correct axes object.

    Returns
    -------
    matplotlib.axes.Axes
        The (possibly new) 3-D axes.
    """
    if not hasattr(ax, "bar3d"):
        ax = fig.add_subplot(projection="3d")
        mpl_axes[idx] = ax
    return ax


def _compute_grid_shape(n_axes: int, row_nums: set, col_nums: set) -> tuple:
    """
    Compute ``(nrow, ncol)`` for the subplot grid.

    Uses recorded row/col indices when available; falls back to a square-ish
    layout otherwise.

    Parameters
    ----------
    n_axes : int
        Number of distinct axes in the figure.
    row_nums, col_nums : set of int
        Row and column indices read from the log ``kargs``.
    """
    if row_nums:
        return max(row_nums) + 1, max(col_nums) + 1
    ncol = math.ceil(math.sqrt(n_axes))
    nrow = math.ceil(n_axes / ncol)
    return nrow, ncol


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def replot_figure(t2vlog_dir: str,
                  verbose: bool = False,
                  force_replot: bool = True) -> None:
    """
    Re-plot all logged series from *t2vlog_dir* and save a merged PNG.

    Each axes is reconstructed from its JSON log files.  The logged data
    points are drawn as red scatter overlays.  The merged image is written to
    ``<t2vlog_dir>/merge_images_<fig_id>.png``.

    Parameters
    ----------
    t2vlog_dir : str
        Path to a directory produced by ``activate_t2v_logging()``.
    verbose : bool
        Print progress messages.
    force_replot : bool
        Overwrite any existing merged image when True.
    """
    # Set a variable to signal that we're in replot mode, so the logging calls
    # in the original code can skip logging to avoid infinite loops.
    set_variable("T2V_ISLOG", False)
    
    all_paths = sorted(glob.glob(os.path.join(t2vlog_dir, "*.json")))

    # Group paths by (fig_id, axes_id)
    figs: dict = defaultdict(lambda: defaultdict(list))
    for path in all_paths:
        stem = os.path.basename(path)
        if stem.startswith(_RESERVED_PREFIXES):
            continue
        parts   = stem.split("_")
        fig_id  = parts[0]
        axes_id = parts[-2]
        figs[fig_id][axes_id].append(path)

    for fig_id, axes in figs.items():
        merged_path = os.path.join(t2vlog_dir, f"merge_images_{fig_id}.png")
        if os.path.exists(merged_path) and not force_replot:
            if verbose:
                print(f"Replot: merged image already exists: {merged_path}, skipping.")
            continue

        if not axes:
            print("No axes found in the directory.")
            continue

        # Determine grid shape from logged ax_row / ax_col
        row_nums: set = set()
        col_nums: set = set()
        for ax_paths in axes.values():
            for path in ax_paths:
                with open(path, encoding="utf-8") as fh:
                    kargs = json.load(fh).get("kargs", {})
                if kargs.get("ax_row") is not None:
                    row_nums.add(kargs["ax_row"])
                if kargs.get("ax_col") is not None:
                    col_nums.add(kargs["ax_col"])

        nrow, ncol = _compute_grid_shape(len(axes), row_nums, col_nums)

        if verbose:
            print(f"Replotting {len(axes)} axes in {nrow} rows × {ncol} cols.")

        fig, mpl_axes = plt.subplots(nrow, ncol,
                                     figsize=(ncol * 8, nrow * 6))
        if isinstance(mpl_axes, matplotlib.axes.Axes):
            mpl_axes = [mpl_axes]
        else:
            mpl_axes = mpl_axes.flatten()

        for axes_id, ax_paths in axes.items():
            for path in ax_paths:
                try:
                    with open(path, encoding="utf-8") as fh:
                        json_data = json.load(fh)

                    args      = json_data["args"]
                    kargs     = json_data["kargs"]
                    axes_row  = kargs.pop("ax_row", 0)
                    axes_col  = kargs.pop("ax_col", 0)
                    axes_idx  = axes_row * ncol + axes_col
                    x_inv     = kargs.pop("x_axis_inverted", False)
                    y_inv     = kargs.pop("y_axis_inverted", False)
                    func_name = json_data["plt_func"]
                    data_series = json_data.get("data_series", [])

                    ax   = mpl_axes[axes_idx]
                    func = choose_axes_func(ax, func_name)

                    # Skip unknown or decoration functions
                    if func is None and not any([
                        func_name.startswith("sns"),
                        func_name.startswith("geo"),
                        func_name.endswith("3d"),
                    ]):
                        print(f"Unknown function: {func_name}")
                        continue
                    if func_name in ("set_xticks", "set_yticks",
                                     "set_xticklabels", "set_yticklabels",
                                     "axvline", "axhline"):
                        continue

                    # ----------------------------------------------------------
                    # Per-function replot logic
                    # ----------------------------------------------------------

                    if func_name == "pie":
                        x = filter_out_invalid_values(data_series[0]["x"])
                        ax.pie(x)

                    elif func_name in ("bar", "barh", "hist"):
                        _try_original(func, args, kargs)
                        for s in data_series:
                            x, y = filter_out_invalid_values(s["x"], s["y"])
                            ax.scatter(x, y, **_OVERLAY)

                    elif func_name == "bar3d":
                        ax = _ensure_3d(ax, fig, mpl_axes, axes_idx)
                        ax.bar3d(*args, **kargs)
                        for s in data_series:
                            x, y, z = filter_out_invalid_values(s["x"], s["y"], s["z"])
                            ax.scatter3D(x, y, z, color="red", s=10, zorder=math.inf)

                    elif func_name == "plot":
                        for s in data_series:
                            xs, ys = filter_out_invalid_values(s["x"], s["y"])
                            if s.get("is_x_datetime"):
                                xs = [matplotlib.dates.num2date(v) for v in xs]
                            if s.get("is_y_datetime"):
                                ys = [matplotlib.dates.num2date(v) for v in ys]
                            ax.plot(xs, ys,
                                    marker=s.get("marker"),
                                    linestyle=s.get("linestyle"),
                                    color=s.get("color"))
                            ax.scatter(xs, ys, **_OVERLAY)

                    elif func_name == "scatter":
                        for s in data_series:
                            sz = s.get("s") or [5] * len(s["x"])
                            x, y, sz = filter_out_invalid_values(s["x"], s["y"], sz)
                            ax.scatter(x, y, color="red", s=sz, zorder=math.inf)

                    elif func_name == "scatter3d":
                        ax = _ensure_3d(ax, fig, mpl_axes, axes_idx)
                        for s in data_series:
                            x, y, z, sz = filter_out_invalid_values(
                                s["x"], s["y"], s["z"], s["s"])
                            ax.scatter(x, y, z, color="red", s=sz, zorder=math.inf)

                    elif func_name in ("fill_between", "fill_betweenx"):
                        for s in data_series:
                            x, y = filter_out_invalid_values(s["x"], s["y"])
                            ax.fill(x, y, alpha=s.get("alpha", 0.5),
                                    color=s.get("color"))
                            ax.scatter(x, y, **_OVERLAY)

                    elif func_name in ("imshow", "pcolormesh", "hist2d", "hexbin"):
                        for s in data_series:
                            x, y, z = filter_out_invalid_values(s["x"], s["y"], s["z"])
                            ax.scatter(x, y, c=z, s=10, zorder=math.inf)

                    elif func_name in ("contour", "contourf"):
                        _try_original(func, args, kargs)
                        for s in data_series:
                            x, y, z = filter_out_invalid_values(s["x"], s["y"], s["z"])
                            z   = np.array(z)
                            lo, hi = s["cmin"], s["cmax"]
                            if lo != hi:
                                z = (z - lo) / (hi - lo)
                            ax.scatter(s["x"], s["y"], s=(z + 0.01) * 50, color="red")

                    elif func_name == "geo":
                        fig.set_size_inches(12, 9)
                        for s in data_series:
                            x, y = filter_out_invalid_values(s["x"], s["y"])
                            patch = plt.Polygon(
                                list(zip(x, y)), closed=True, fill=True,
                                edgecolor="k", linewidth=0.5,
                                facecolor=s.get("color"),
                            )
                            ax.add_patch(patch)
                        x_lims = kargs.get("x_lims")
                        y_lims = kargs.get("y_lims")
                        if x_lims is not None:
                            ax.set_xlim(min(x_lims[0], -200), max(x_lims[1], 200))
                        if y_lims is not None:
                            ax.set_ylim(min(y_lims[0], -100), max(y_lims[1], 100))

                    elif func_name in ("violinplot", "violin"):
                        _try_original(func, args, kargs)
                        for s in data_series:
                            pts = s["data"]
                            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                                       **_OVERLAY)

                    elif func_name in ("sns_plot_strips", "sns_plot_swarms"):
                        for s in data_series:
                            ax.scatter(x=s["x"], y=s["y"],
                                       color=s["color"], s=5, zorder=math.inf)

                    elif func_name in ("bxp", "sns_plot_boxens",
                                       "sns_plot_boxes", "sns_plot_violins"):
                        for s in data_series:
                            vert = s.get("orientation", "vertical") != "horizontal"
                            pos_key = "x" if vert else "y"
                            q_key   = "y" if vert else "x"
                            pos       = s[pos_key][0]
                            quantiles = s[q_key]
                            bxpstats  = [{
                                "whislo": quantiles[0], "q1": quantiles[1],
                                "med":    quantiles[2], "q3": quantiles[3],
                                "whishi": quantiles[4],
                            }]
                            ax.bxp(bxpstats, positions=[pos], vert=vert,
                                   showfliers=False,
                                   boxprops={"color": s["color"]})
                            xs = [pos] * len(quantiles) if vert else quantiles
                            ys = quantiles if vert else [pos] * len(quantiles)
                            ax.scatter(xs, ys, **_OVERLAY)

                    elif func_name == "errorbar":
                        _try_original(func, args, kargs)
                        for s in data_series:
                            ax.scatter(s["x"], s["y"], **_OVERLAY)

                    elif func_name in ("stem", "stairs"):
                        _try_original(func, args, kargs)
                        for s in data_series:
                            ax.scatter(s["x"], s["y"], **_OVERLAY)

                    elif func_name == "plot_surface3d":
                        ax = _ensure_3d(ax, fig, mpl_axes, axes_idx)
                        _try_original(ax.plot_surface, args, kargs)
                        for s in data_series:
                            ax.scatter(s["x"], s["y"], s["z"],
                                       color="red", s=1, zorder=-math.inf)

                    else:
                        _try_original(func, args, kargs)

                    if x_inv:
                        mpl_axes[axes_idx].invert_xaxis()
                    if y_inv:
                        mpl_axes[axes_idx].invert_yaxis()

                    mpl_axes[axes_idx].figure.canvas.draw()

                except Exception as e:
                    traceback.print_exc()
                    print(f"Replot: error in axes {axes_id} from {path}: {e}")

        try:
            fig.tight_layout()
            fig.savefig(merged_path, bbox_inches="tight", dpi=300)
        except Exception as e:
            print(f"Replot: error saving {merged_path}: {e}")
        finally:
            pass
            # plt.close("all")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log_dir_glob = (
        "/Users/nngu0448/Documents/data/T2V-Phase2-Experiments"
        "/z_new_analysis_data_copy/arxiv-v3-all_human-nlr_codex-gpt5_1"
        "/r1_mail_209_1_1/*/r1_mail_209_1_1_log"
    )
    for log_dir in tqdm(sorted(glob.glob(log_dir_glob))):
        replot_figure(log_dir, verbose=False, force_replot=True)
