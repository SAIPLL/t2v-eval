"""
Batch T2V logging runner.

Discovers prediction and ground-truth Python scripts under an experiment
directory, injects the T2V logging activation code, executes each script in a
subprocess, and records the logging outcome.

Usage
-----
    python t2v_batch_log.py \\
        --experiment_dir /data/experiment \\
        --data_rel_dir   batch_01 \\
        --py_env         python3 \\
        --timeout        300 \\
        --use_cache \\
        --replot
"""

import glob
import os
import subprocess
import time
import argparse

import pandas as pd
from tqdm import tqdm

from t2v_eval.logging import (
    LOGGING_ACTIVATION_CODE,
    LoggingState,
    setup_log_dir,
)
from t2v_eval.logging.replot import replot_figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prepare_py_file_for_logging(py_path_in: str,
                                 py_path_out: str,
                                 log_dir: str) -> str | None:
    """
    Insert the T2V logging activation line into a Python file.

    The activation line is inserted immediately before the line that contains
    ``##START VISUALISATION CODE`` (case-insensitive).

    Parameters
    ----------
    py_path_in  : path to the original Python script.
    py_path_out : path where the instrumented script will be written.
    log_dir     : directory that ``activate_t2v_logging`` will write JSON files to.

    Returns
    -------
    str or None
        *py_path_out* on success; ``None`` if the marker line was not found.
    """
    with open(py_path_in, encoding="utf-8") as fh:
        lines = fh.read().split("\n")

    marker_idx = next(
        (i for i, ln in enumerate(lines)
         if "##start visualisation code" in ln.lower()),
        None,
    )
    if marker_idx is None:
        print(f"[WARNING] '##START VISUALISATION CODE' not found in: {py_path_in}")
        return None

    lines.insert(marker_idx, LOGGING_ACTIVATION_CODE.format(log_dir=log_dir))
    with open(py_path_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    return py_path_out


def is_valid_log_dir(log_dir: str, verbose: bool = False) -> bool:
    """
    Return True if *log_dir* exists and contains more than one file.

    A freshly created log directory always contains ``variables.json``, so
    a count of ≤ 1 means nothing was actually plotted.
    """
    if not os.path.exists(log_dir):
        if verbose:
            print(f"  Log dir does not exist: {log_dir}")
        return False
    n = sum(len(files) for _, _, files in os.walk(log_dir))
    if n <= 1:
        if verbose:
            print(f"  Log dir has no data files: {log_dir}")
        return False
    if verbose:
        print(f"  Log dir valid ({n} files): {log_dir}")
    return True


def execute_and_log_data_py_path(py_path: str,
                                  py_env: str = "python3",
                                  timeout: int = 300,
                                  use_cache: bool = False,
                                  replot: bool = False,
                                  verbose: bool = False) -> tuple[LoggingState, str]:
    """
    Execute a Python script with T2V logging injected and return the result.

    The script must contain the ``##START VISUALISATION CODE`` marker.

    Parameters
    ----------
    py_path   : absolute path to the Python script.
    py_env    : Python interpreter to use.
    timeout   : maximum execution time in seconds.
    use_cache : skip execution if a valid log directory already exists.
    replot    : render a merged PNG from the log directory after execution.
    verbose   : print stdout/stderr from the subprocess.

    Returns
    -------
    (LoggingState, log_dir)
    """
    assert "/t2v_pred/" in py_path or "/t2v_gt/" in py_path, (
        f"Script path must contain /t2v_pred/ or /t2v_gt/: {py_path}"
    )

    py_log_dir  = py_path.rsplit(".", maxsplit=1)[0] + "_log"
    py_path_out = py_log_dir + ".py"

    # --- cache hit ---
    if use_cache and is_valid_log_dir(py_log_dir, verbose=verbose):
        return LoggingState.HAVING_DATA, py_log_dir

    # --- prepare log directory ---
    setup_log_dir(py_log_dir)

    # --- inject logging activation ---
    if prepare_py_file_for_logging(py_path, py_path_out, py_log_dir) is None:
        return LoggingState.BAD_INPUT, py_log_dir

    # --- execute ---
    try:
        result = subprocess.run(
            [py_env, py_path_out],
            cwd=os.path.dirname(py_path_out),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if verbose:
            if result.stdout:
                print("  STDOUT:\n" + result.stdout)
            if result.stderr:
                print("  STDERR:\n" + result.stderr)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {py_path_out}")
        return LoggingState.NO_DATA, py_log_dir
    except Exception as exc:
        print(f"  [ERROR] executing {py_path_out}: {exc}")
        return LoggingState.NO_DATA, py_log_dir

    # --- validate & optionally replot ---
    if is_valid_log_dir(py_log_dir, verbose=verbose):
        if replot:
            try:
                replot_figure(py_log_dir, verbose=verbose)
            except Exception as exc:
                print(f"  [ERROR] replotting {py_log_dir}: {exc}")
        return LoggingState.HAVING_DATA, py_log_dir

    return LoggingState.NO_DATA, py_log_dir


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_logging_batch(experiment_dir: str,
                      data_rel_dir: str,
                      py_env: str = "python3",
                      timeout: int = 300,
                      use_cache: bool = False,
                      replot: bool = False,
                      verbose: bool = False) -> pd.DataFrame:
    """
    Discover all prediction scripts under *experiment_dir/data_rel_dir*, run
    T2V logging on each, and mirror the same operation on the paired GT script.

    Prediction scripts are located via::

        {experiment_dir}/{data_rel_dir}/*/t2v_pred/*.vis-request.txt

    The corresponding ``.py`` file is expected beside the ``.vis-request.txt``
    file.  The GT script lives at the same relative path under ``t2v_gt/``.

    Results are saved to ``logs/t2v_logging/{data_rel_dir}_logging.csv``.

    Returns
    -------
    pd.DataFrame with one row per visualisation request.
    """
    glob_path = f"{experiment_dir}/{data_rel_dir}/*/t2v_pred/*.vis-request.txt"
    vsr_paths = sorted(glob.glob(glob_path))
    print(f"Found {len(vsr_paths)} vis-request files under: {glob_path}\n")

    rows = []
    for vsr_path in tqdm(vsr_paths, desc="Logging scripts"):

        # --- prediction script ---
        pred_py_path = vsr_path.replace(".vis-request.txt", ".py")
        if os.path.exists(pred_py_path):
            t0 = time.time()
            pred_state, pred_log_dir = execute_and_log_data_py_path(
                pred_py_path,
                py_env=py_env,
                timeout=timeout,
                use_cache=use_cache,
                replot=replot,
                verbose=verbose,
            )
            pred_duration = time.time() - t0
        else:
            print(f"  [MISSING] pred script: {pred_py_path}")
            pred_state, pred_log_dir, pred_duration = LoggingState.BAD_INPUT, "", -1.0

        # --- ground-truth script (always cached once logged) ---
        gt_py_path = pred_py_path.replace("/t2v_pred/", "/t2v_gt/")
        if os.path.exists(gt_py_path):
            t0 = time.time()
            gt_state, gt_log_dir = execute_and_log_data_py_path(
                gt_py_path,
                py_env=py_env,
                timeout=timeout,
                use_cache=True,   # GT is fixed — always use cache when available
                replot=replot,
                verbose=verbose,
            )
            gt_duration = time.time() - t0
        else:
            print(f"  [MISSING] gt script: {gt_py_path}")
            gt_state, gt_log_dir, gt_duration = LoggingState.BAD_INPUT, "", -1.0

        rows.append({
            "vsr_path":          vsr_path,
            "pred_py_path":      pred_py_path,
            "pred_logging_state":str(pred_state.value),
            "pred_log_dir":      pred_log_dir,
            "pred_duration_s":   round(pred_duration, 2),
            "gt_py_path":        gt_py_path,
            "gt_logging_state":  str(gt_state.value),
            "gt_log_dir":        gt_log_dir,
            "gt_duration_s":     round(gt_duration, 2),
        })

    df = pd.DataFrame(rows)
    log_file = f"logs/t2v_logging/{data_rel_dir}_logging.csv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    df.to_csv(log_file, index=False)

    # --- summary ---
    pred_ok = (df["pred_logging_state"] == LoggingState.HAVING_DATA.value).sum()
    gt_ok   = (df["gt_logging_state"]   == LoggingState.HAVING_DATA.value).sum()
    print(f"\nLogging complete.")
    print(f"  Pred: {pred_ok}/{len(df)} succeeded")
    print(f"  GT:   {gt_ok}/{len(df)} succeeded")
    print(f"  Results saved → {log_file}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run T2V logging on prediction and ground-truth scripts."
    )
    parser.add_argument("--experiment_dir", type=str, required=True,
                        help="Root directory of the experiment.")
    parser.add_argument("--data_rel_dir",   type=str, required=True,
                        help="Relative path to the data batch folder.")
    parser.add_argument("--py_env",         type=str, default="python3",
                        help="Python interpreter to use (default: python3).")
    parser.add_argument("--timeout",        type=int, default=300,
                        help="Per-script execution timeout in seconds (default: 300).")
    parser.add_argument("--use_cache",      action="store_true",
                        help="Skip scripts whose log directory already exists.")
    parser.add_argument("--replot",         action="store_true",
                        help="Render a merged PNG overlay after each script.")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print subprocess stdout/stderr and debug info.")
    args = parser.parse_args()

    run_logging_batch(
        experiment_dir = args.experiment_dir,
        data_rel_dir   = args.data_rel_dir,
        py_env         = args.py_env,
        timeout        = args.timeout,
        use_cache      = args.use_cache,
        replot         = args.replot,
        verbose        = args.verbose,
    )


if __name__ == "__main__":
    main()
