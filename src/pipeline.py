"""
The core backtesting pipeline.
"""
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
from rich.console import Console

from src.config import Config
from src.reporting import (
    generate_run_manifest,
    generate_summary_json,
    generate_summary_markdown,
    generate_trade_ledger_csv,
)
from src.types import RunContext, RunProposals, RunResult, Trade

__all__ = ["run_backtest"]


def _get_git_hash() -> str:
    """Returns the current git hash, or 'unknown' if not a git repo."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _generate_mock_result(config: Config) -> RunResult:
    """
    Creates a mock RunResult for demonstration purposes.
    In a real scenario, this would be the output of a proper backtest.
    """
    trades = [
        Trade(
            symbol="RELIANCE.NS",
            entry_date=date(2023, 1, 5),
            exit_date=date(2023, 1, 10),
            entry_price=2500.0,
            exit_price=2550.0,
            sample_type="IS",
        ),
        Trade(
            symbol="TCS.NS",
            entry_date=date(2023, 2, 3),
            exit_date=date(2023, 2, 8),
            entry_price=3400.0,
            exit_price=3350.0,
            sample_type="OOS",
        ),
    ]
    signals = pd.DataFrame({"date": [date(2023, 1, 5), date(2023, 2, 3)], "signal": [1, -1]})
    context = RunContext(config=config, git_hash=_get_git_hash(), data_hash="mock_data_hash")
    proposals = RunProposals(signals=signals, trades=trades)
    metrics = {"total_return_pct": 0.01, "sharpe_ratio": 1.5, "num_trades": 2}
    return RunResult(context=context, proposals=proposals, metrics=metrics)


# impure
def run_backtest(config: Config, console: Console) -> None:
    """
    Runs the main backtesting pipeline and generates reports.
    #impure: This function has side effects (I/O).
    """
    console.print(f"Backtest started for run: [bold]{config.run.name}[/bold]")

    # This is where the actual backtesting logic would go.
    # For now, we generate a mock result.
    result = _generate_mock_result(config)

    # Ensure the output directory exists.
    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all reports.
    generate_trade_ledger_csv(result, output_dir)
    generate_summary_json(result, output_dir)
    generate_summary_markdown(result, output_dir)
    generate_run_manifest(result, output_dir)

    console.print(f"Reports saved to: [cyan]{output_dir.resolve()}[/cyan]")
