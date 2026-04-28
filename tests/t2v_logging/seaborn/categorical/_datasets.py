"""Synthetic datasets for seaborn categorical tests (no network required)."""

import numpy as np
import pandas as pd


def make_tips(seed: int = 42) -> pd.DataFrame:
    """Synthetic tips: total_bill, tip, sex, smoker, day, time, size."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "total_bill": rng.uniform(3, 50, 200),
        "tip":        rng.uniform(1, 10, 200),
        "sex":        list(rng.choice(["Male", "Female"], 200)),
        "smoker":     list(rng.choice(["Yes", "No"], 200)),
        "day":        list(rng.choice(["Thur", "Fri", "Sat", "Sun"], 200)),
        "time":       list(rng.choice(["Lunch", "Dinner"], 200)),
        "size":       list(rng.integers(1, 7, 200).astype(int)),
    })


def make_titanic(seed: int = 42) -> pd.DataFrame:
    """Synthetic titanic: sex, survived, class, age, deck, fare, embark_town."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sex":         list(rng.choice(["male", "female"], 200)),
        "survived":    list(rng.integers(0, 2, 200).astype(float)),
        "class":       list(rng.choice(["First", "Second", "Third"], 200)),
        "age":         list(rng.uniform(1, 80, 200)),
        "deck":        list(rng.choice(["A", "B", "C", "D", "E"], 200)),
        "fare":        list(rng.uniform(5, 500, 200)),
        "embark_town": list(rng.choice(
            ["Southampton", "Cherbourg", "Queenstown"], 200)),
    })
