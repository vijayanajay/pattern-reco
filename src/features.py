"""
Feature engineering: returns, gaps, rolling stats, etc.

Functions in this module should be pure and operate on a single DataFrame.
"""

import pandas as pd

__all__ = ["add_features"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds core features required for backtesting and analysis.

    - Turnover: Close * Volume
    - returns: Percentage change in closing price.
    - gap_pct: Overnight gap percentage from previous close to current open.

    Args:
        df: Input DataFrame with 'Open', 'Close', 'Volume'.

    Returns:
        A new DataFrame with the added feature columns.
    """
    if not all(col in df.columns for col in ["Open", "Close", "Volume"]):
        raise ValueError("Input DataFrame must contain 'Open', 'Close', and 'Volume' columns.")

    # Work on a copy to avoid side effects
    df_out = df.copy()

    # Simple, direct calculations as per design.
    df_out["Turnover"] = df_out["Close"] * df_out["Volume"]
    df_out["returns"] = df_out["Close"].pct_change()
    df_out["gap_pct"] = (df_out["Open"] - df_out["Close"].shift(1)) / df_out["Close"].shift(1)

    return df_out
