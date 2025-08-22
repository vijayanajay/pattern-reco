"""
Walk-forward validation, execution, and portfolio logic using vectorbt.

This module is the core of the backtesting engine. It takes feature-rich
price data and signals, and uses the `vectorbt` library to simulate
trade execution and portfolio performance.
"""

import pandas as pd
import numpy as np
import vectorbt as vbt
from rich.console import Console

from src.config import Config, DetectorConfig
from src.detectors import generate_signals

__all__ = ["run"]


def _prepare_vbt_data(
    processed_data: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares data for vectorbt by combining individual symbol dataframes
    into multi-column dataframes for prices and signals.
    """
    # Use 'Open' for entries and 'Close' for exits as a simplification.
    # The design specifies next-day open, which vbt handles via `entry_prices`
    open_prices = pd.concat(
        [df["Open"] for df in processed_data.values()],
        axis=1,
        keys=processed_data.keys(),
    )
    close_prices = pd.concat(
        [df["Close"] for df in processed_data.values()],
        axis=1,
        keys=processed_data.keys(),
    )
    return open_prices, close_prices


def run(
    config: Config,
    processed_data: dict[str, pd.DataFrame],
    console: Console,
) -> vbt.Portfolio:
    """
    Main entry point for the backtesting engine.

    This prepares data for vectorbt, generates signals, and runs the backtest
    using the specified configuration. For the MVP, it uses the first parameter
    set from the config.
    """
    console.print("Preparing data for vectorbt...")
    open_prices, close_prices = _prepare_vbt_data(processed_data)

    # TODO: Implement full walk-forward with parameter grid search.
    # For now, use the first parameter set from the config.
    detector_cfg = config.detector
    window = detector_cfg.window_range[0]
    k_low = detector_cfg.k_low_range[0]
    console.print(f"Using detector params: window={window}, k_low={k_low}")

    console.print("Generating signals for all symbols...")
    all_signals = []
    for symbol, df in processed_data.items():
        signals = generate_signals(df, window=window, k_low=k_low)
        all_signals.append(signals.rename(symbol))

    entry_signals = pd.concat(all_signals, axis=1)

    # To simulate next-day entry, we shift the signals by one day.
    shifted_entries = entry_signals.vbt.fshift(1)

    console.print("Running backtest with vectorbt...")

    portfolio = vbt.Portfolio.from_signals(
        close=close_prices,  # Use close price for transactions and valuation
        entries=shifted_entries,
        exits=np.nan, # Time-based exits are handled by `freq`
        freq=f"{config.detector.max_hold}D",
        fees=config.execution.fees_bps / 10000.0,
        init_cash=1e9,
    )

    console.print("Backtest complete.")
    return portfolio
