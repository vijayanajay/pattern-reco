"""
Feature engineering: returns, gaps, rolling stats, etc.
"""
from typing import Dict

import pandas as pd

__all__ = ["add_features"]


def add_features(
    data: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """
    Adds engineered features to the raw data.
    For now, it only adds 'Turnover'.
    """
    for symbol in data:
        df = data[symbol]
        # Rule [H-3]: Prefer simple, direct calculations.
        df["Turnover"] = df["Close"] * df["Volume"]
        data[symbol] = df
    return data
