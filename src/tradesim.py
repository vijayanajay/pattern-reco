"""
Minimalist and efficient trade simulation for strategy evaluation.
"""
import pandas as pd
import numpy as np

__all__ = ["calculate_returns"]


def calculate_returns(
    data: pd.DataFrame, signals: pd.Series, max_hold: int
) -> pd.Series:
    """
    Calculates returns for a simple trading strategy based on entry signals.

    This function is optimized for performance using vectorized operations. It
    assumes entry on the day after the signal at the 'Open' price, and exit
    `max_hold` days later at the 'Close' price.

    Args:
        data: A DataFrame containing at least 'Open' and 'Close' price columns,
              with a DatetimeIndex.
        signals: A boolean Series of the same length as `data`, where `True`
                 indicates a trade entry signal on that day.
        max_hold: The number of trading days to hold the position.

    Returns:
        A pandas Series containing the calculated returns for each valid trade.
        Returns an empty Series if no valid trades can be executed.
    """
    if not all(c in data.columns for c in ["Open", "Close"]):
        raise ValueError("Input `data` DataFrame must contain 'Open' and 'Close' columns.")

    # Find the integer locations of all entry signals.
    signal_locs = np.where(signals)[0]

    # Determine entry and exit locations. Entry is the day after the signal.
    entry_locs = signal_locs + 1
    exit_locs = entry_locs + max_hold

    # Filter out trades where entry or exit would be beyond the data range.
    valid_mask = exit_locs < len(data)
    entry_locs = entry_locs[valid_mask]
    exit_locs = exit_locs[valid_mask]

    if len(entry_locs) == 0:
        return pd.Series(dtype=np.float64)

    # Retrieve entry and exit prices using the valid locations.
    entry_prices = data["Open"].iloc[entry_locs].values
    exit_prices = data["Close"].iloc[exit_locs].values

    # Calculate returns and filter out any trades with missing price data.
    returns = (exit_prices - entry_prices) / entry_prices
    return pd.Series(returns[~np.isnan(returns)])
