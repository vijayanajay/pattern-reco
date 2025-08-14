"""
CLI entry point for the pattern-reco application.
"""
from pathlib import Path

import typer
from rich.console import Console

from src.config import Config, load_config
from src.data import discover_symbols, fetch_and_snapshot
from src.pipeline import run_pipeline

# Follows rule [H-18], console is created once and passed down.
# Log to stderr to separate from potential data output to stdout.
app = typer.Typer(pretty_exceptions_show_locals=False, help="Anomaly & Pattern Detection for Indian Stocks.")
console = Console(stderr=True)


def _load_config_or_exit(config_path: Path) -> Config:
    """Helper to load config and exit on failure."""
    try:
        return load_config(config_path)
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def run(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to the YAML configuration file.", exists=True
    ),
):
    """Execute the backtest pipeline based on the given configuration."""
    config = _load_config_or_exit(config_path)
    # Call the new unified pipeline
    run_pipeline(config, console)
    console.print("[bold green]Run command finished.[/bold green]")


@app.command(name="refresh-data")
def refresh_data(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to the YAML configuration file.", exists=True
    ),
):
    """
    Refresh data snapshots from the source (e.g., yfinance).
    """
    config = _load_config_or_exit(config_path)
    console.print("Starting data refresh...")

    symbols_to_refresh = discover_symbols(config)
    if not symbols_to_refresh:
        # Fallback to the include_symbols list if no snapshots exist
        symbols_to_refresh = config.universe.get("include_symbols", [])
        if symbols_to_refresh:
            console.print("No existing snapshots found. Performing initial download for symbols in config.")
        else:
            console.print("[yellow]Warning: No symbols to refresh.[/yellow]")
            console.print("No snapshots found and 'universe.include_symbols' is empty.")
            raise typer.Exit()
    else:
        console.print(f"Found {len(symbols_to_refresh)} existing symbols. Refreshing them.")

    # New fetch_and_snapshot returns a list of failures
    failed_symbols = fetch_and_snapshot(symbols_to_refresh, config)

    if failed_symbols:
        console.print(f"[bold yellow]Warning:[/bold yellow] Failed to fetch data for {len(failed_symbols)} symbols:")
        for symbol in sorted(failed_symbols):
            console.print(f" - {symbol}")
        # According to design, this is a soft warning, not a hard fail.

    console.print("[bold green]Data refresh completed.[/bold green]")


if __name__ == "__main__":
    app()
