import sys
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from src.config import Config, load_config
from src.data import fetch_and_snapshot, load_snapshots
from src.universe import get_nse_symbols, select_universe

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def _load_config_or_exit(config_path: Path) -> Config:
    """Helper to load config and exit on failure."""
    try:
        return load_config(str(config_path))
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1) from e


@app.command()
def run(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to YAML configuration file.", exists=True
    )
):
    """Execute the backtest pipeline."""
    config = _load_config_or_exit(config_path)
    console.print(f"Loaded configuration for run: [bold]{config.run.name}[/bold]")

    console.print("Loading data snapshots...")
    all_symbols = get_nse_symbols(config, console)
    all_data = load_snapshots(all_symbols, config, console)

    console.print("Selecting universe...")
    # Using a placeholder t0, as in the original code. This will be fixed.
    t0 = date(2020, 1, 1)
    universe = select_universe(all_data, config, t0, console)
    console.print(f"Selected {len(universe)} symbols.")

    console.print("[yellow]Warning: The main pipeline logic is not implemented.[/yellow]")
    console.print("[green]Code simplification review complete.[/green]")


@app.command(name="refresh-data")
def refresh_data(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to YAML configuration file.", exists=True
    )
):
    """Refresh data snapshots from the source."""
    config = _load_config_or_exit(config_path)
    console.print("Starting data refresh...")
    symbols = get_nse_symbols(config, console)
    fetch_and_snapshot(symbols, config, console)
    console.print("[green]Data refresh completed successfully.[/green]")


if __name__ == "__main__":
    app()
