"""
CLI entry point for the pattern-reco application.
"""
from pathlib import Path

import typer
from rich.console import Console

from src.backtest import PipelineError, run_pipeline
from src.config import Config, load_config
from src.data import refresh_market_data

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
) -> None:
    """Execute the backtest pipeline based on the given configuration."""
    config = _load_config_or_exit(config_path)
    try:
        run_pipeline(config, console)
    except PipelineError as e:
        console.print(f"[bold red]Pipeline Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1)

    console.print("\n[bold green]Backtest run finished successfully.[/bold green]")


@app.command(name="refresh-data")
def refresh_data(
    config_path: Path = typer.Option(
        ..., "--config", "-c", help="Path to the YAML configuration file.", exists=True
    ),
) -> None:
    """Refresh data snapshots from the source (e.g., yfinance)."""
    config = _load_config_or_exit(config_path)
    try:
        refresh_market_data(config, console)
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
