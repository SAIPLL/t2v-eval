"""
Chart-type accuracy metric for T2V evaluation.

Entry point: ``calculate_chart_type_accuracy(pred_log_dir, gt_log_dir, ...)``
"""

from .metric_utils import read_t2v_log_dir


def _get_chart_types(fig_data: dict) -> set[str]:
    """Return the set of all plt_func values present in fig_data."""
    return {
        series["plt_func"]
        for axis_data in fig_data.values()
        for series in axis_data
    }


def calculate_chart_type_accuracy(pred_log_dir: str,
                                   gt_log_dir: str,
                                   down_sampling: bool = True,
                                   down_sampling_max_length: int = 5_000,
                                   down_sampling_method: str = "random",
                                   dedup: bool = False,
                                   summarize_before: bool = False,
                                   summarize_after: bool = False,
                                   print_points: int = 0,
                                   verbose: bool = False) -> bool:
    """
    Check whether the predicted visualization uses all chart types present
    in the ground truth.

    A sample is considered correct (returns ``True``) when the set of chart
    types in the prediction is a superset of the ground-truth chart types.

    Parameters
    ----------
    pred_log_dir / gt_log_dir
        Paths to log directories produced by ``activate_t2v_logging()``.
    (remaining parameters are forwarded to ``read_t2v_log_dir``)

    Returns
    -------
    bool  ``True`` if all GT chart types are covered by the prediction.
    """
    _kwargs = dict(
        dedup=dedup,
        down_sampling=down_sampling,
        down_sampling_max_length=down_sampling_max_length,
        down_sampling_method=down_sampling_method,
        verbose=verbose,
        summarize_before=summarize_before,
        summarize_after=summarize_after,
        print_points=print_points,
    )
    pred_fig_data = read_t2v_log_dir(pred_log_dir, **_kwargs)
    gt_fig_data   = read_t2v_log_dir(gt_log_dir,   **_kwargs)

    pred_types = _get_chart_types(pred_fig_data)
    gt_types   = _get_chart_types(gt_fig_data)

    return gt_types.issubset(pred_types)
