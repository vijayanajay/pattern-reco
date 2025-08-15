"""
Generating output reports from a vectorbt backtest.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

import vectorbt as vbt
from rich.console import Console

from src.config import Config

__all__ = ["generate_all_reports"]


def _to_json_serializable(data):
    """Recursively converts non-serializable types in a dictionary."""
    if isinstance(data, dict):
        return {k: _to_json_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_json_serializable(i) for i in data]
    if isinstance(data, (Path, pd.Timestamp, pd.Timedelta)):
        return str(data)
    if pd.isna(data) or data is None:
        return None
    # Convert numpy types to native Python types
    if isinstance(data, (np.integer, np.int64)):
        return int(data)
    if isinstance(data, (np.floating, np.float64)):
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)
    return data


# impure
def _generate_trade_ledger_csv(portfolio: vbt.Portfolio, output_dir: Path) -> None:
    """Generates a CSV file with all trade details."""
    # vectorbt's trades record is comprehensive.
    trades = portfolio.trades.records
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(output_dir / "trade_ledger.csv", index=False)


# impure
def _generate_summary_json(portfolio: vbt.Portfolio, config: Config, output_dir: Path) -> None:
    """Generates a JSON file with summary metrics."""
    # portfolio.stats() returns a Series with many metrics.
    stats = portfolio.stats()
    summary = {
        "run_name": config.run.name,
        "detector_params": {
            "window": config.detector.window_range[0],
            "k_low": config.detector.k_low_range[0],
        },
        "metrics": _to_json_serializable(stats.to_dict()),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


# impure
def _generate_summary_markdown(portfolio: vbt.Portfolio, config: Config, output_dir: Path) -> None:
    """Generates a Markdown file with a human-readable summary."""
    stats = portfolio.stats()
    md = f"# Backtest Summary: {config.run.name}\n\n"
    md += "## Key Metrics\n\n"

    key_metrics = [
        "Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio",
        "Win Rate [%]", "Total Trades", "Avg Winning Trade [%]", "Avg Losing Trade [%]"
    ]

    for metric in key_metrics:
        if metric in stats:
            value = stats[metric]
            md += f"- **{metric}**: {value:.2f}\n"

    (output_dir / "summary.md").write_text(md)


# impure
def generate_all_reports(
    config: Config,
    portfolio: vbt.Portfolio,
    run_dir: Path,
    console: Console,
) -> None:
    """
    Orchestrates the generation of all output reports.
    #impure: Writes to the filesystem.
    """
    if not isinstance(portfolio, vbt.Portfolio):
        console.print("[bold red]Error: Invalid backtest result. Cannot generate reports.[/bold red]")
        return

    formats = config.reporting.output_formats

    if "csv" in formats:
        console.print("Generating trade ledger CSV...")
        _generate_trade_ledger_csv(portfolio, run_dir)

    if "json" in formats:
        console.print("Generating summary JSON...")
        _generate_summary_json(portfolio, config, run_dir)

    if "markdown" in formats:
        console.print("Generating summary Markdown...")
        _generate_summary_markdown(portfolio, config, run_dir)

    # Optional: Generate plots if specified
    if config.reporting.generate_plots:
        console.print("Generating plots...")
        try:
            fig = portfolio.plot()
            fig.write_image(run_dir / "portfolio_summary.png")
        except Exception as e:
            console.print(f"[yellow]Warning: Plot generation failed. {e}[/yellow]")
            console.print("[yellow]You may need to install 'plotly' and 'kaleido'.[/yellow]")

    console.print("All reports generated.")
