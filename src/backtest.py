"""
Main pipeline orchestration and backtesting engine.

This module orchestrates the entire backtesting pipeline, from data loading
and feature engineering to execution and reporting. It uses vectorbt for
the core backtest simulation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt
from rich.console import Console

from src.config import Config
from src.data import load_snapshots, select_universe
from src.detectors import generate_signals
from src.features import add_features
from src.reporting import generate_all_reports

__all__ = ["run_pipeline", "PipelineError"]


class PipelineError(Exception):
    """Custom exception for pipeline failures."""


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


# impure
def _run_vbt_backtest(
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
        exits=np.nan,  # Time-based exits are handled by `freq`
        freq=f"{config.detector.max_hold}D",
        fees=config.execution.fees_bps / 10000.0,
        init_cash=1e9,
    )

    console.print("Backtest complete.")
    return portfolio


# impure
def run_pipeline(config: Config, console: Console) -> None:
    """
    Execute the full backtest pipeline from data loading to report generation.
    """
    # Step 1: Setup - Universe and Data Loading
    console.rule("[bold]1. Setting up Run[/bold]")
    universe = select_universe(config, console)
    if not universe:
        raise PipelineError("Universe is empty. Check config `universe` settings.")

    snapshots = load_snapshots(universe, config, console)
    if not snapshots:
        raise PipelineError("No data loaded. Run `refresh-data` or check universe.")

    # Step 2: Feature Processing
    console.rule("[bold]2. Processing Features[/bold]")
    processed_data = {symbol: add_features(df) for symbol, df in snapshots.items()}
    console.print("Feature processing complete.")

    # Step 3: Backtest Execution
    console.rule("[bold]3. Executing Backtest[/bold]")
    results = _run_vbt_backtest(config, processed_data, console)
    console.print("Backtest execution complete.")

    # Step 4: Reporting
    console.rule("[bold]4. Generating Reports[/bold]")
    run_dir = Path(config.run.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Run artifacts will be saved to: [cyan]{run_dir}[/cyan]")
    generate_all_reports(config, results, run_dir, console)
    console.print("Reporting complete.")
