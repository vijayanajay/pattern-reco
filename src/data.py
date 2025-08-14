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
    # Using attribute access with the new Pydantic models
    return config.data.snapshot_dir / f"{config.data.source}_{config.data.interval}"


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

    failed_symbols = []
    for symbol in symbols:
        try:
            # Use yf.download for robustness and cleaner logs.
            data = yf.download(
                tickers=symbol,
                start=config.data.start_date,
                end=config.data.end_date,
                interval=config.data.interval,
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

        df = pd.read_parquet(parquet_path)
        # As per rule [H-7], ensure required columns exist.
        required_cols = {"Close", "Volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Data for {symbol} is missing required columns: {required_cols - set(df.columns)}")

        # Rule [H-3]: Prefer simple, direct calculations.
        df["Turnover"] = df["Close"] * df["Volume"]
        loaded_data[symbol] = df

    return loaded_data
