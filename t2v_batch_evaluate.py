"""
Batch T2V evaluation runner.

Computes displayed-data similarity and chart-type accuracy for every
(prediction, ground-truth) log-directory pair in an experiment batch.

Usage
-----
    python t2v_batch_evaluate.py \\
        --experiment_dir /data/experiment \\
        --data_rel_dir   batch_01 \\
        --use_cache
"""

import os
import glob
import pandas as pd
import argparse

from tqdm import tqdm
from t2v_eval.evaluation.ddata_similarity import calculate_similarity_score
from t2v_eval.evaluation.chart_type_accuracy import calculate_chart_type_accuracy

_LOG_ROOT_DSIM = "logs/metrics-scores/ddata_similarity"
_LOG_ROOT_CTA  = "logs/metrics-scores/chart_type_accuracy"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_logdir_pairs(experiment_dir: str,
                           data_rel_dir: str,
                           data_rel_dir_2: str | None = None) -> list[tuple[str, str]]:
    """
    Discover all (pred_log_dir, gt_log_dir) pairs under the given experiment
    directory.  Optionally combines two data-relative directories.
    """
    gt_glob = f"{experiment_dir}/{data_rel_dir}/*/t2v_gt/*_log"
    gt_dirs = sorted(glob.glob(gt_glob))
    print(f"Found {len(gt_dirs)} GT log dirs in '{data_rel_dir}'.")

    if data_rel_dir_2 is not None:
        gt_glob_2 = f"{experiment_dir}/{data_rel_dir_2}/*/t2v_gt/*_log"
        gt_dirs_2 = sorted(glob.glob(gt_glob_2))
        gt_dirs  += gt_dirs_2
        print(f"Found {len(gt_dirs_2)} GT log dirs in '{data_rel_dir_2}'.")

    assert len(gt_dirs) > 0, (
        f"No GT log directories found under glob: {gt_glob}"
    )

    pairs = []
    for gt in gt_dirs:
        pred = gt.replace("/t2v_gt/", "/t2v_pred/")
        if not os.path.exists(pred):
            print(f"Warning: pred log dir not found: {pred}")
        pairs.append((pred, gt))

    print(f"Collected {len(pairs)} (pred, gt) pairs.\n")
    return pairs


def _csv_path(log_root: str, data_rel_dir: str,
              log_file_name: str | None, suffix: str) -> str:
    """Build the output CSV file path."""
    stem = log_file_name if log_file_name else data_rel_dir
    return os.path.join(log_root, f"{stem}_{suffix}.csv")


# ---------------------------------------------------------------------------
# Displayed data similarity
# ---------------------------------------------------------------------------

def evaluate_ddata_similarity(experiment_dir: str,
                              data_rel_dir: str,
                              data_rel_dir_2: str | None = None,
                              log_file_name: str | None = None,
                              use_cache: bool = True,
                              verbose: bool = False) -> tuple[float, pd.DataFrame]:
    """
    Compute the displayed-data similarity score for every (pred, gt) pair.

    Results are saved to ``logs/metrics-scores/ddata_similarity/``.

    Returns
    -------
    (avg_score, df)
        Average similarity score and the per-sample DataFrame.
    """
    pairs = _collect_logdir_pairs(experiment_dir, data_rel_dir, data_rel_dir_2)

    rows = []
    for pred_log_dir, gt_log_dir in tqdm(pairs, desc="Data similarity"):
        pred_exists = os.path.exists(pred_log_dir)
        if not pred_exists:
            print(f"  Skipping (pred missing): {pred_log_dir}")
            score = 0.0
        else:
            try:
                score = calculate_similarity_score(
                    pred_log_dir,
                    gt_log_dir,
                    down_sampling=True,
                    down_sampling_max_length=10000,
                    down_sampling_method="random",
                    dedup=False,
                    summarize_before=False,
                    summarize_after=False,
                    use_cache=use_cache,
                    save_results=True,
                    verbose=verbose,
                )
            except Exception as exc:
                print(f"  Error: {exc}")
                score = None

        rows.append({
            "pred_log_dir":        pred_log_dir,
            "pred_log_dir_exists": pred_exists,
            "gt_log_dir":          gt_log_dir,
            "gt_log_dir_exists":   os.path.exists(gt_log_dir),
            "similarity_score":    score,
        })

    df  = pd.DataFrame(rows)
    csv = _csv_path(_LOG_ROOT_DSIM, data_rel_dir, log_file_name, "sim_score")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"Saved data-similarity results → {csv}")

    avg = df["similarity_score"].mean()
    print(f"Average displayed-data similarity ({len(df)} samples): {avg:.4f}\n")
    return avg, df


# ---------------------------------------------------------------------------
# Chart type accuracy
# ---------------------------------------------------------------------------

def evaluate_chart_type_accuracy(experiment_dir: str,
                                  data_rel_dir: str,
                                  data_rel_dir_2: str | None = None,
                                  log_file_name: str | None = None,
                                  use_cache: bool = True,
                                  verbose: bool = False) -> tuple[float, pd.DataFrame]:
    """
    Compute the chart-type accuracy for every (pred, gt) pair.

    A sample is correct (1) when the prediction's chart-type set is a
    superset of the ground truth's chart-type set.

    Results are saved to ``logs/metrics-scores/chart_type_accuracy/``.

    Returns
    -------
    (accuracy, df)
        Fraction of correct samples and the per-sample DataFrame.
    """
    pairs = _collect_logdir_pairs(experiment_dir, data_rel_dir, data_rel_dir_2)

    rows = []
    for pred_log_dir, gt_log_dir in tqdm(pairs, desc="Chart type accuracy"):
        pred_exists = os.path.exists(pred_log_dir)
        if not pred_exists:
            print(f"  Skipping (pred missing): {pred_log_dir}")
            correct = False
        else:
            try:
                correct = calculate_chart_type_accuracy(
                    pred_log_dir,
                    gt_log_dir,
                    verbose=verbose,
                )
            except Exception as exc:
                print(f"  Error: {exc}")
                correct = None

        rows.append({
            "pred_log_dir":        pred_log_dir,
            "pred_log_dir_exists": pred_exists,
            "gt_log_dir":          gt_log_dir,
            "gt_log_dir_exists":   os.path.exists(gt_log_dir),
            "chart_type_correct":  correct,
        })

    df  = pd.DataFrame(rows)
    csv = _csv_path(_LOG_ROOT_CTA, data_rel_dir, log_file_name, "chart_type_accuracy")
    os.makedirs(os.path.dirname(csv), exist_ok=True)
    df.to_csv(csv, index=False)
    print(f"Saved chart-type-accuracy results → {csv}")

    accuracy = df["chart_type_correct"].mean()
    print(f"Chart-type accuracy ({len(df)} samples): {accuracy:.4f}\n")
    return accuracy, df


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_all(experiment_dir: str,
                 data_rel_dir: str,
                 data_rel_dir_2: str | None = None,
                 log_file_name: str | None = None,
                 use_cache: bool = True,
                 verbose: bool = False) -> dict:
    """
    Run both metrics and print a combined summary.

    Returns
    -------
    dict with keys ``similarity_score`` and ``chart_type_accuracy``.
    """
    print("=" * 60)
    print("Displayed Data Similarity")
    print("=" * 60)
    avg_sim, _ = evaluate_ddata_similarity(
        experiment_dir, data_rel_dir, data_rel_dir_2,
        log_file_name, use_cache, verbose,
    )

    print("=" * 60)
    print("Chart Type Accuracy")
    print("=" * 60)
    avg_cta, _ = evaluate_chart_type_accuracy(
        experiment_dir, data_rel_dir, data_rel_dir_2,
        log_file_name, use_cache, verbose,
    )

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Displayed data similarity : {avg_sim:.4f}")
    print(f"  Chart type accuracy       : {avg_cta:.4f}")
    print("=" * 60)

    return {
        "similarity_score":    avg_sim,
        "chart_type_accuracy": avg_cta,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate T2V predictions against ground-truth log directories."
    )
    parser.add_argument("--experiment_dir",  type=str, required=True,
                        help="Root directory of the experiment.")
    parser.add_argument("--data_rel_dir",    type=str, required=True,
                        help="Relative path to the first data batch folder.")
    parser.add_argument("--data_rel_dir_2",  type=str, default=None,
                        help="Relative path to a second data batch folder (optional).")
    parser.add_argument("--log_file_name",   type=str, default=None,
                        help="Stem for output CSV filenames (defaults to data_rel_dir).")
    parser.add_argument("--metric",          type=str, default="all",
                        choices=["all", "similarity", "chart_type"],
                        help="Which metric to compute (default: all).")
    parser.add_argument("--use_cache",       action="store_true",
                        help="Use cached evaluation results where available.")
    parser.add_argument("--verbose",         action="store_true",
                        help="Print debug information during evaluation.")
    args = parser.parse_args()

    kwargs = dict(
        experiment_dir  = args.experiment_dir,
        data_rel_dir    = args.data_rel_dir,
        data_rel_dir_2  = args.data_rel_dir_2,
        log_file_name   = args.log_file_name,
        use_cache       = args.use_cache,
        verbose         = args.verbose,
    )

    if args.metric == "similarity":
        evaluate_ddata_similarity(**kwargs)
    elif args.metric == "chart_type":
        evaluate_chart_type_accuracy(**kwargs)
    else:
        evaluate_all(**kwargs)


if __name__ == "__main__":
    main()
