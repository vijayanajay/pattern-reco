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

    @staticmethod
    def add_gap_z_signal(df: pd.DataFrame, window: int, k_low: float) -> pd.DataFrame:
        """
        Generates Gap-Z signals and scores.

        A signal is generated when the z-score of the gap percentage falls
        below a given threshold (`k_low`). The score is the z-score itself,
        which can be used for ranking signals.

        Args:
            df: DataFrame with OHLC data.
            window: The rolling window for z-score calculation.
            k_low: The z-score threshold for generating a signal.

        Returns:
            DataFrame with added 'signal_gapz' and 'score_gapz' columns.
        """
        zscore_col = f"gap_pct_zscore_{window}"

        # Chain feature calculations for a clean, readable pipeline
        df_featured = df.pipe(FeatureEngine.add_gap_percentage).pipe(
            FeatureEngine.add_rolling_zscore, column="gap_pct", window=window
        )

        # Generate signal and score
        signal = (df_featured[zscore_col] < k_low).fillna(False)
        score = df_featured[zscore_col]

        return df_featured.assign(signal_gapz=signal, score_gapz=score)
