"""
Signal detection logic.

For the MVP, this contains the Gap-Z detector. The functions are pure,
taking a DataFrame and parameters, and returning a signal Series.
"""

import pandas as pd

__all__ = ["generate_signals"]


def _rolling_zscore(series: pd.Series, window: int, min_periods: int = None) -> pd.Series:
    """
    Calculates the rolling z-score of a time series.
    z = (value - rolling_mean) / rolling_std
    """
    if min_periods is None:
        min_periods = window // 2  # A reasonable default
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    # Avoid division by zero; replace std=0 with a small epsilon or NaN
    rolling_std = rolling_std.replace(0, pd.NA)
    return (series - rolling_mean) / rolling_std


def generate_signals(df: pd.DataFrame, window: int, k_low: float) -> pd.Series:
    """
    Generates entry signals based on the Gap-Z strategy.

    The signal is generated when the z-score of the overnight gap percentage
    falls below a specified negative threshold.

    Args:
        df: DataFrame must contain a 'gap_pct' column.
        window: The rolling window for the z-score calculation.
        k_low: The negative z-score threshold to trigger a long signal.

    Returns:
        A boolean Series where True indicates a trade entry signal.
    """
    if "gap_pct" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'gap_pct' column from feature engineering.")

    # A check to ensure the threshold is negative, as this is a long-only strategy
    # for negative gaps.
    if k_low >= 0:
        raise ValueError("k_low threshold must be a negative value for this strategy.")

    z_scores = _rolling_zscore(df["gap_pct"], window=window)

    # The signal is True when the z-score is below the negative threshold (e.g., < -2.0)
    signals = z_scores < k_low

    return signals
