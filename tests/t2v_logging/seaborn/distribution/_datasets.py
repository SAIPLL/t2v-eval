"""Synthetic datasets for seaborn distribution tests (no network required)."""

import numpy as np
import pandas as pd


def make_penguins(seed: int = 42) -> pd.DataFrame:
    """Synthetic penguins: flipper_length_mm, bill_length/depth, species, sex."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "flipper_length_mm": np.concatenate([
            rng.normal(190, 7, 70),
            rng.normal(200, 6, 65),
            rng.normal(215, 8, 65),
        ]),
        "bill_length_mm": np.concatenate([
            rng.normal(39, 3, 70),
            rng.normal(48, 3, 65),
            rng.normal(46, 3, 65),
        ]),
        "bill_depth_mm": np.concatenate([
            rng.normal(18, 1, 70),
            rng.normal(17, 1, 65),
            rng.normal(15, 1, 65),
        ]),
        "species": ["Adelie"] * 70 + ["Chinstrap"] * 65 + ["Gentoo"] * 65,
        "sex":     list(rng.choice(["Male", "Female"], 200)),
    })


def make_tips(seed: int = 42) -> pd.DataFrame:
    """Synthetic tips: total_bill, size, day."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "total_bill": rng.uniform(3, 50, 100),
        "size":       rng.integers(1, 7, 100).astype(int),
        "day":        list(rng.choice(["Sun", "Mon", "Tue", "Wed"], 100)),
    })
