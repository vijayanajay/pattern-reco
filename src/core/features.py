"""
Stateless feature engineering functions.
"""
from typing import Optional
import numpy as np
import pandas as pd

__all__ = [
    "add_daily_returns",
    "add_gap_percentage",
    "add_rolling_zscore",
    "add_gap_z_signal",
]


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily returns from 'Close' prices."""
    if "Close" not in df.columns:
        raise ValueError("DataFrame must have a 'Close' column.")
    return df.assign(daily_return=df["Close"].pct_change())


def add_gap_percentage(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the overnight gap percentage: (Open - PrevClose) / PrevClose."""
    if not all(c in df.columns for c in ["Open", "Close"]):
        raise ValueError("DataFrame must have 'Open' and 'Close' columns.")
    return df.assign(
        gap_pct=(df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    )


def add_rolling_zscore(
    df: pd.DataFrame, column: str, window: int, min_periods: Optional[int] = None
) -> pd.DataFrame:
    """Computes a rolling z-score for a given column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")

    min_periods = min_periods or int(window * 0.9)

    rolling = df[column].rolling(window=window, min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std().replace(0, np.nan)  # Avoid division by zero

    zscore_col = f"{column}_zscore_{window}"
    return df.assign(**{zscore_col: (df[column] - mean) / std})


def add_gap_z_signal(df: pd.DataFrame, window: int, k_low: float) -> pd.DataFrame:
    """
    Generates Gap-Z signals and scores based on a z-score threshold.
    A signal is generated when the z-score of the gap percentage falls below k_low.
    """
    df_gaps = add_gap_percentage(df)
    df_zscore = add_rolling_zscore(df_gaps, column="gap_pct", window=window)

    zscore_col = f"gap_pct_zscore_{window}"
    signal = (df_zscore[zscore_col] < k_low).fillna(False)

    return df_zscore.assign(signal_gapz=signal, score_gapz=df_zscore[zscore_col])
