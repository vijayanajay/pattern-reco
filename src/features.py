"""
Feature engineering components.
Clean, minimal, and efficient feature calculation functions.
"""
import pandas as pd
import numpy as np

__all__ = ["FeatureEngine"]


class FeatureEngine:
    """
    A collection of stateless, chainable functions for feature engineering.
    This design promotes simplicity, testability, and composability.
    Each function takes a DataFrame and returns a new DataFrame with the added feature.
    """

    @staticmethod
    def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates daily returns from 'Close' prices."""
        if "Close" not in df.columns:
            raise ValueError("DataFrame must have 'Close' column")
        return df.assign(daily_return=df["Close"].pct_change())

    @staticmethod
    def add_gap_percentage(df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the overnight gap percentage."""
        if not all(c in df.columns for c in ["Open", "Close"]):
            raise ValueError("DataFrame must have 'Open' and 'Close' columns")
        return df.assign(
            gap_pct=(df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
        )

    @staticmethod
    def add_rolling_zscore(
        df: pd.DataFrame, column: str, window: int
    ) -> pd.DataFrame:
        """Computes a rolling z-score using mean and standard deviation."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not in DataFrame")
        rolling = df[column].rolling(window=window)
        mean = rolling.mean()
        std = rolling.std()
        return df.assign(
            **{f"{column}_zscore_{window}": (df[column] - mean) / std.replace(0, np.nan)}
        )
