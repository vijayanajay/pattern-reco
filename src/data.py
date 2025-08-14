"""
Data fetching and snapshot management from yfinance.
"""
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

from src.config import Config

__all__ = ["fetch_and_snapshot", "load_snapshots", "discover_symbols"]


def _get_snapshot_dir(config: Config) -> Path:
    """Constructs the snapshot directory path from config."""
    # Using dictionary access now
    data_cfg = config.data
    return Path(data_cfg["snapshot_dir"]) / f"{data_cfg['source']}_{data_cfg['interval']}"


def discover_symbols(config: Config) -> List[str]:
    """Discovers all available symbols by scanning the snapshot directory."""
    snapshot_dir = _get_snapshot_dir(config)
    if not snapshot_dir.exists():
        return []
    return sorted([p.stem for p in snapshot_dir.glob("*.parquet")])


# impure
def fetch_and_snapshot(symbols: List[str], config: Config) -> List[str]:
    """
    Fetch data from yfinance and save to parquet snapshots.
    Returns a list of symbols that failed to download.
    #impure: Accesses network and filesystem.
    """
    snapshot_dir = _get_snapshot_dir(config)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = config.data

    failed_symbols = []
    for symbol in symbols:
        try:
            # Use yf.download for robustness and cleaner logs.
            data = yf.download(
                tickers=symbol,
                start=data_cfg["start_date"],
                end=data_cfg["end_date"],
                interval=data_cfg["interval"],
                auto_adjust=True,
                prepost=False,
                actions=False,
                progress=False,  # Keep logs clean
            )
            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            parquet_path = snapshot_dir / f"{symbol}.parquet"
            data.to_parquet(parquet_path, engine="pyarrow")

        except (IOError, ConnectionError, ValueError):
            # Instead of logging, we collect failures for the caller to handle.
            failed_symbols.append(symbol)

    return failed_symbols


# impure
def load_snapshots(symbols: List[str], config: Config) -> Dict[str, pd.DataFrame]:
    """
    Load existing data snapshots for a list of symbols.
    Raises FileNotFoundError if any requested symbol's snapshot is missing.
    #impure: Reads from the filesystem.
    """
    snapshot_dir = _get_snapshot_dir(config)
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    loaded_data = {}
    for symbol in symbols:
        parquet_path = snapshot_dir / f"{symbol}.parquet"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing snapshot for symbol: {symbol} at {parquet_path}")
        loaded_data[symbol] = pd.read_parquet(parquet_path)

    return loaded_data
