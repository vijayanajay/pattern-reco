"""
The core backtesting pipeline.

This module provides a single, unified entry point for running a backtest.
As of now, it only performs data loading and validation. The core
walk-forward, signal generation, and execution logic is not yet implemented.
"""

from rich.console import Console

from src.config import Config
from src.data import discover_symbols, load_snapshots
from src.features import add_features

__all__ = ["run_pipeline"]


# impure
def run_pipeline(config: Config, console: Console) -> None:
    """
    Runs the main backtesting pipeline.
    #impure: This function will have side effects (I/O, etc.).
    """
    console.print(f"Pipeline started for run: [bold]{config.run.name}[/bold]")

    # 1. Load data
    console.print("Loading data snapshots...")
    symbols = discover_symbols(config)
    if not symbols:
        console.print("[bold red]Error: No data snapshots found.[/bold red]")
        console.print("Please run the 'refresh-data' command first.")
        return

    all_data = load_snapshots(symbols, config)
    console.print(f"Successfully loaded data for {len(symbols)} symbols.")

    # 2. Add features
    console.print("Adding features...")
    all_data = add_features(all_data)

    # 3. Pipeline Execution (Placeholder)
    console.print("\n[bold yellow]Warning: Core pipeline logic is not implemented.[/bold yellow]")
    console.print("This script currently only loads data and does not perform a backtest.")
    console.print("Exiting.")
