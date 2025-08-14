"""
The core backtesting pipeline.

This module replaces the deleted `universe`, `walk_forward`, and `execution`
modules, providing a single, unified entry point for running a backtest.
"""

from rich.console import Console

from src.config import Config

__all__ = ["run_pipeline"]


# impure
def run_pipeline(config: Config, console: Console) -> None:
    """
    Runs the main backtesting pipeline.
    #impure: This function will have side effects (I/O, etc.).
    """
    console.print(f"Pipeline started for run: [bold]{config.run['name']}[/bold]")
    console.print("[yellow]Warning: Main pipeline logic is a placeholder.[/yellow]")

    # In the future, this function will:
    # 1. Load data using functions from `src.data`.
    # 2. Select the stock universe.
    # 3. Create walk-forward splits.
    # 4. For each split:
    #    a. Fit detector parameters on in-sample data.
    #    b. Generate signals on out-of-sample data.
    #    c. Simulate execution and costs.
    # 5. Aggregate results and generate reports.

    console.print("Pipeline placeholder finished.")
