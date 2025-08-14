"""
The core backtesting pipeline.

This module provides a single, unified entry point for running a backtest.
"""

from typing import Dict, List, NamedTuple, Set

import numpy as np
import pandas as pd
from rich.console import Console

from src.config import Config
from src.data import discover_symbols, load_snapshots

__all__ = ["run_pipeline"]


class Position(NamedTuple):
    """
    Represents an open position.
    """

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    size: float  # In currency value (e.g., INR)


def _select_positions(
    signals: pd.DataFrame,
    all_data: Dict[str, pd.DataFrame],
    portfolio_cfg: Dict,
    open_positions: Set[str],
) -> List[Position]:
    """
    Selects new positions based on signals and portfolio constraints.

    Args:
        signals: DataFrame with 'symbol', 'date', and 'signal' columns.
        all_data: Dict mapping symbols to their historical data.
        portfolio_cfg: The portfolio section of the configuration.
        open_positions: A set of symbols for currently open positions.

    Returns:
        A list of new `Position` objects to be opened.
    """
    max_concurrent = portfolio_cfg["max_concurrent"]
    available_slots = max_concurrent - len(open_positions)
    if available_slots <= 0 or signals.empty:
        return []

    # Calculate median turnover for ranking
    median_turnover = {
        symbol: data["Turnover"].median() for symbol, data in all_data.items()
    }
    signals["median_turnover"] = signals["symbol"].map(median_turnover)

    # Apply re-entry lockout (rule 5.4)
    if portfolio_cfg.get("reentry_lockout", True):
        candidate_signals = signals[~signals["symbol"].isin(open_positions)].copy()
    else:
        candidate_signals = signals.copy()

    # Deterministic sorting (rule 5.1)
    # 1. Signal extremity (descending)
    # 2. Median turnover (descending)
    # 3. Ticker symbol (ascending)
    candidate_signals = candidate_signals.sort_values(
        by=["signal", "median_turnover", "symbol"],
        ascending=[False, False, True],
    )

    # Select top N candidates
    selected_candidates = candidate_signals.head(available_slots)
    if selected_candidates.empty:
        return []

    # Create Position objects (rules 5.3, 5.5)
    new_positions = []
    position_size = portfolio_cfg["position_size"]
    for _, row in selected_candidates.iterrows():
        symbol = row["symbol"]
        entry_date = row["date"]
        # Assume entry price is the open price on the signal day.
        # This is a placeholder; a real system would use the next day's open.
        entry_price = all_data[symbol].loc[entry_date]["Open"]
        new_positions.append(
            Position(
                symbol=symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                size=position_size,
            )
        )

    return new_positions


# impure
def run_pipeline(config: Config, console: Console) -> None:
    """
    Runs the main backtesting pipeline.
    #impure: This function will have side effects (I/O, etc.).
    """
    console.print(f"Pipeline started for run: [bold]{config.run['name']}[/bold]")

    # 1. Load data
    symbols = discover_symbols(config)
    if not symbols:
        console.print("[bold red]No data snapshots found. Run 'refresh-data' first.[/bold red]")
        return
    all_data = load_snapshots(symbols, config)
    console.print(f"Loaded data for {len(symbols)} symbols.")

    # --- Placeholder for walk-forward loop and signal generation ---
    # In a real pipeline, we would loop through time.
    # For now, we simulate signal generation for a single day.
    t0 = pd.to_datetime(config.run["t0"])
    console.print(f"\nSimulating for a single date: {t0.date()}")

    # Create dummy signals for all symbols present in the data at t0
    dummy_signals = []
    for symbol, df in all_data.items():
        if t0 in df.index:
            # Generate a random signal value for demonstration
            dummy_signals.append(
                {
                    "date": t0,
                    "symbol": symbol,
                    "signal": np.random.rand(),
                }
            )
    signal_df = pd.DataFrame(dummy_signals)

    if signal_df.empty:
        console.print(f"[yellow]No data available for any symbol on {t0.date()}. Cannot generate signals.[/yellow]")
        return

    console.print(f"Generated {len(signal_df)} dummy signals.")

    # --- Portfolio Management ---
    # Simulate an empty set of open positions for this single run
    open_positions: Set[str] = set()
    console.print(f"Initial open positions: {len(open_positions)}")

    # Call the position selection logic
    new_positions = _select_positions(
        signals=signal_df,
        all_data=all_data,
        portfolio_cfg=config.portfolio,
        open_positions=open_positions,
    )

    console.print(f"\nSelected {len(new_positions)} new positions:")
    for pos in new_positions:
        console.print(f"- {pos.symbol} | Size: {pos.size} | Entry: {pos.entry_price:.2f} on {pos.entry_date.date()}")

    console.print("\nPipeline placeholder finished.")
