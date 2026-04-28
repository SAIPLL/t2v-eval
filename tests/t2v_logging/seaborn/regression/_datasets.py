"""Synthetic datasets for seaborn regression tests (no network required)."""

import numpy as np
import pandas as pd


def make_tips(seed: int = 42) -> pd.DataFrame:
    """Synthetic tips: total_bill, tip, smoker, time, sex, size."""
    rng = np.random.default_rng(seed)
    n   = 100
    df  = pd.DataFrame({
        "total_bill": rng.uniform(3, 50, n),
        "tip":        rng.uniform(1, 10, n),
        "smoker":     list(rng.choice(["Yes", "No"], n)),
        "time":       list(rng.choice(["Lunch", "Dinner"], n)),
        "sex":        list(rng.choice(["Male", "Female"], n)),
        "size":       list(rng.integers(1, 7, n).astype(int)),
    })
    df["big_tip"] = (df["tip"] / df["total_bill"] > 0.15).astype(float)
    return df


def make_anscombe() -> dict:
    """
    Anscombe's quartet — four datasets with near-identical statistics
    but very different regression behaviour.
    """
    return {
        "I": pd.DataFrame({
            "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96,
                  7.24, 4.26, 10.84, 4.82, 5.68],
        }),
        "II": pd.DataFrame({
            "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10,
                  6.13, 3.10, 9.13, 7.26, 4.74],
        }),
        "III": pd.DataFrame({
            "x": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
            "y": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84,
                  6.08, 5.39, 8.15, 6.42, 5.73],
        }),
    }
