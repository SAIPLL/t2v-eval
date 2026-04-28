"""Synthetic datasets for seaborn axis-grid tests (no network required)."""

import numpy as np
import pandas as pd


def make_tips(seed: int = 42, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "total_bill": rng.uniform(3, 50, n),
        "tip":        rng.uniform(1, 10, n),
        "sex":        list(rng.choice(["Male", "Female"], n)),
        "smoker":     list(rng.choice(["Yes", "No"], n)),
        "day":        list(rng.choice(["Thur", "Fri", "Sat", "Sun"], n)),
        "time":       list(rng.choice(["Lunch", "Dinner"], n)),
        "size":       list(rng.integers(1, 7, n).astype(int)),
    })


def make_iris(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sepal_length": rng.uniform(4, 8, 120),
        "sepal_width":  rng.uniform(2, 5, 120),
        "petal_length": rng.uniform(1, 7, 120),
        "petal_width":  rng.uniform(0, 3, 120),
        "species":      ["setosa"] * 40 + ["versicolor"] * 40 + ["virginica"] * 40,
    })


def make_attend(seed: int = 42) -> pd.DataFrame:
    """12 subjects × 3 solution levels."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "subject":   list(range(1, 13)) * 3,
        "solutions": [1, 2, 3] * 12,
        "score":     rng.uniform(0, 10, 36),
    })
