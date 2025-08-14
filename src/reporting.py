"""
Generating output reports, metrics, and trade ledgers.
"""
import json
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.types import RunResult, Trade

__all__ = [
    "calculate_trade_returns",
    "generate_trade_ledger_csv",
    "generate_summary_json",
    "generate_summary_markdown",
    "generate_run_manifest",
]


def calculate_trade_returns(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([t.dict() for t in trades])
    df["return_pct"] = (df["exit_price"] / df["entry_price"]) - 1
    df["duration_days"] = (pd.to_datetime(df["exit_date"]) - pd.to_datetime(df["entry_date"])).dt.days
    return df


# impure
def generate_trade_ledger_csv(result: RunResult, output_dir: Path) -> None:
    """Generates a CSV file with all trade details."""
    trades_df = calculate_trade_returns(result.proposals.trades)
    if not trades_df.empty:
        trades_df.to_csv(output_dir / "trade_ledger.csv", index=False)


# impure
def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def generate_summary_json(result: RunResult, output_dir: Path) -> None:
    """Generates a JSON file with summary metrics."""
    summary = {
        "run_name": result.context.config.run.name,
        "metrics": result.metrics,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=4, default=json_serializer)


# impure
def generate_summary_markdown(result: RunResult, output_dir: Path) -> None:
    """Generates a Markdown file with a human-readable summary."""
    md = f"# Run Summary: {result.context.config.run.name}\n\n"
    md += "## Metrics\n"
    for key, value in result.metrics.items():
        md += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
    (output_dir / "summary.md").write_text(md)


# impure
def generate_run_manifest(result: RunResult, output_dir: Path) -> None:
    """Creates a manifest file with run metadata."""
    manifest = {
        "run_name": result.context.config.run.name,
        "git_hash": result.context.git_hash,
        "data_hash": result.context.data_hash,
        "config": result.context.config.dict(),
    }
    with (output_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=4, default=json_serializer)
