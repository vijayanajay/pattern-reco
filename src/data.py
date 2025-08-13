"""
Data fetching and snapshot management from yfinance.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf
from rich.console import Console

from src.config import Config

__all__ = ["fetch_and_snapshot", "load_snapshots", "discover_symbols"]


def _get_snapshot_dir(config: Config) -> Path:
    """Constructs the snapshot directory path from config."""
    return config.data.snapshot_dir / f"{config.data.source}_{config.data.interval}"


def discover_symbols(config: Config) -> List[str]:
    """Discovers all available symbols by scanning the snapshot directory."""
    snapshot_dir = _get_snapshot_dir(config)
    if not snapshot_dir.exists():
        return []

    return [p.stem for p in snapshot_dir.glob("*.parquet")]


# impure
def fetch_and_snapshot(symbols: List[str], config: Config, console: Console) -> None:
    """
    Fetch data from yfinance and save to parquet snapshots.
    Logs warnings for symbols that fail to download.
    """
    snapshot_dir = _get_snapshot_dir(config)
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

        except (IOError, ConnectionError, ValueError) as e:
            console.log(f"[red]Warning: Failed to fetch data for {symbol}: {e}[/red]")
            failed_symbols.append(symbol)

    if failed_symbols:
        console.log(f"[yellow]Warning: Failed to fetch data for {len(failed_symbols)} symbols.[/yellow]")


# impure
def load_snapshots(symbols: List[str], config: Config, console: Console) -> Dict[str, pd.DataFrame]:
    """
    Load existing data snapshots for a list of symbols.
    """
    snapshot_dir = _get_snapshot_dir(config)
    console.log(f"Loading {len(symbols)} snapshots from: {snapshot_dir}")

    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Snapshot directory not found: {snapshot_dir}. "
            "Run 'refresh-data' to fetch data."
        )

    loaded_data = {}
    for symbol in symbols:
        parquet_path = snapshot_dir / f"{symbol}.parquet"
        if not parquet_path.exists():
            # This is not an error, just means we don't have data for this symbol.
            # The universe selection logic will handle it.
            continue
        loaded_data[symbol] = pd.read_parquet(parquet_path)

    console.log(f"Successfully loaded {len(loaded_data)} snapshots.")
    return loaded_data
