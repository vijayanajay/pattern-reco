"""
Data fetching and snapshot management from yfinance.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf
from rich.console import Console

from src.config import Config

__all__ = ["fetch_and_snapshot", "load_snapshots"]


def fetch_and_snapshot(symbols: List[str], config: Config, console: Console) -> None:  # impure
    """
    Fetch data from yfinance and save to parquet snapshots.
    Logs warnings for symbols that fail to download.
    """
    snapshot_dir = Path(config.data.snapshot_dir) / f"{config.data.source}_{config.data.interval}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Using snapshot directory: {snapshot_dir}")

    failed_symbols = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=config.data.start_date,
                end=config.data.end_date,
                interval=config.data.interval,
                auto_adjust=True,
                prepost=False,
                actions=False,
            )

            if data.empty:
                console.log(f"[yellow]Warning: No data returned for {symbol}. Skipping.[/yellow]")
                failed_symbols.append(symbol)
                continue

            parquet_path = snapshot_dir / f"{symbol}.parquet"
            data.to_parquet(parquet_path, engine="pyarrow")

        except Exception as e:
            console.log(f"[red]Warning: Failed to fetch data for {symbol}: {e}[/red]")
            failed_symbols.append(symbol)
            continue

    if failed_symbols:
        console.log(f"[yellow]Warning: Failed to fetch data for {len(failed_symbols)} symbols.[/yellow]")


def load_snapshots(symbols: List[str], config: Config, console: Console) -> Dict[str, pd.DataFrame]:  # impure
    """
    Load existing data snapshots for a list of symbols.
    """
    snapshot_dir = Path(config.data.snapshot_dir) / f"{config.data.source}_{config.data.interval}"
    console.log(f"Loading snapshots from: {snapshot_dir}")

    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Snapshot directory not found: {snapshot_dir}. "
            "Run with refresh=True to fetch data."
        )

    loaded_data = {}
    for symbol in symbols:
        parquet_path = snapshot_dir / f"{symbol}.parquet"
        if not parquet_path.exists():
            # This is not an error, just means we don't have data for this symbol.
            # The caller can decide how to handle it.
            continue
        loaded_data[symbol] = pd.read_parquet(parquet_path)

    console.log(f"Successfully loaded {len(loaded_data)} snapshots.")
    return loaded_data
