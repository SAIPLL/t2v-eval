"""
Synthetic datasets for seaborn relational tests.

These replicate the structure of the seaborn tutorial datasets (tips, fmri)
without requiring network access to seaborn's CDN.
"""

import numpy as np
import pandas as pd


def make_tips(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic tips dataset: total_bill, tip, smoker, time, size."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "total_bill": rng.uniform(3, 50, n),
        "tip":        rng.uniform(1, 10, n),
        "smoker":     rng.choice(["Yes", "No"], n),
        "time":       rng.choice(["Lunch", "Dinner"], n),
        "size":       rng.integers(1, 7, n).astype(int),
    })


def make_fmri(n_subjects: int = 6, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic fmri dataset: subject, timepoint, event, region, signal.

    Repeated measures: each subject × event × region combination has one
    signal value per timepoint (0–17).
    """
    rng = np.random.default_rng(seed)
    timepoints = range(18)
    rows = []
    for subj in range(n_subjects):
        for event in ("stim", "cue"):
            for region in ("frontal", "parietal"):
                for tp in timepoints:
                    rows.append({
                        "subject":   f"s{subj}",
                        "timepoint": tp,
                        "event":     event,
                        "region":    region,
                        "signal":    rng.standard_normal(),
                    })
    return pd.DataFrame(rows)
