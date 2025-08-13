import sys
from pathlib import Path

import typer
from rich.console import Console

from src.config import Config, load_config
from src.data import discover_symbols, fetch_and_snapshot
from src.universe import select_universe

app = typer.Typer(pretty_exceptions_show_locals=False)
console = Console()


def _load_config_or_exit(config_path: Path) -> Config:
    """Helper to load config and exit on failure."""
    try:
        # Note: load_config now expects a Path object
        return load_config(config_path)
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

    console.print("Selecting universe...")
    universe = select_universe(config, console)
    console.print(f"Selected {len(universe)} symbols.")

    console.print("[yellow]Warning: The main pipeline logic is not implemented.[/yellow]")
    console.print("[green]Code simplification review complete.[/green]")


@app.command(name="refresh-data")
def refresh_data(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to YAML configuration file.", exists=True
    )
):
    """
    Refresh data snapshots from the source.

    If snapshots already exist, it refreshes data for all discovered symbols.
    If no snapshots exist, it uses the 'include_symbols' list from the config
    to perform an initial download.
    """
    config = _load_config_or_exit(config_path)
    console.print("Starting data refresh...")

    symbols_to_refresh = discover_symbols(config)
    if symbols_to_refresh:
        console.log(f"Found {len(symbols_to_refresh)} existing symbols. Refreshing them.")
    elif config.universe.include_symbols:
        symbols_to_refresh = config.universe.include_symbols
        console.log("No existing symbols found. Performing initial download for symbols in config.")
    else:
        console.print("[yellow]Warning: No symbols to refresh.[/yellow]")
        console.print("No snapshots found and 'universe.include_symbols' is empty in the config.")
        raise typer.Exit()

    fetch_and_snapshot(symbols_to_refresh, config, console)
    console.print("[green]Data refresh completed successfully.[/green]")


if __name__ == "__main__":
    app()
