"""
CLI entry point for the pattern-reco application.
"""
from pathlib import Path

import typer
from rich.console import Console

from src.config import Config, load_config
from src.data import discover_symbols, fetch_and_snapshot, select_universe, load_snapshots
from src.features import add_features
from src.backtest import run as run_the_backtest
from src.reporting import generate_all_reports

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

    try:
        # Step 1: Setup - Universe and Data Loading
        console.rule("[bold]1. Setting up Run[/bold]")
        universe = select_universe(config, console)
        if not universe:
            console.print("[bold red]Error: Universe is empty. Exiting.[/bold red]")
            raise typer.Exit(1)

        snapshots = load_snapshots(universe, config, console)
        if not snapshots:
            console.print("[bold red]Error: No data loaded. Exiting.[/bold red]")
            raise typer.Exit(1)

        # Step 2: Feature Processing
        console.rule("[bold]2. Processing Features[/bold]")
        processed_data = {symbol: add_features(df) for symbol, df in snapshots.items()}
        console.print("Feature processing complete.")

        # Step 3: Backtest Execution
        console.rule("[bold]3. Executing Backtest[/bold]")
        results = run_the_backtest(config, processed_data, console)
        console.print("Backtest execution complete.")

        # Step 4: Reporting
        console.rule("[bold]4. Generating Reports[/bold]")
        run_dir = Path(config.run.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Run artifacts will be saved to: [cyan]{run_dir}[/cyan]")
        generate_all_reports(config, results, run_dir, console)
        console.print("Reporting complete.")

    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred during the run:[/bold red] {e}")
        # In a real scenario, you might want more specific error handling and logging.
        raise typer.Exit(code=1)

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
        symbols_to_refresh = config.universe.include_symbols
        if symbols_to_refresh:
            console.print("No existing snapshots found. Performing initial download for symbols in config.")
        else:
            console.print("[yellow]Warning: No symbols to refresh.[/yellow]")
            console.print("No snapshots found and 'universe.include_symbols' is empty.")
            raise typer.Exit()
    else:
        console.print(f"Found {len(symbols_to_refresh)} existing symbols. Refreshing them.")

    failed_symbols = fetch_and_snapshot(symbols_to_refresh, config)

    if failed_symbols:
        console.print(f"[bold yellow]Warning:[/bold yellow] Failed to fetch data for {len(failed_symbols)} symbols:")
        for symbol in sorted(failed_symbols):
            console.print(f" - {symbol}")

    console.print("[bold green]Data refresh completed.[/bold green]")


if __name__ == "__main__":
    app()
