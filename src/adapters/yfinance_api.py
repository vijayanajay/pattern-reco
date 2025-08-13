"""
Data fetching and snapshot management from yfinance.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

__all__ = ["fetch_and_snapshot", "load_snapshots"]

log = logging.getLogger(__name__)


def fetch_and_snapshot(symbols: List[str], config: Dict[str, Any]) -> None:  # impure
    """
    Fetch data from yfinance and save to parquet snapshots.
    Logs warnings for symbols that fail to download.
    """
    data_config = config.get("data", {})
    snapshot_dir_base = data_config.get("snapshot_dir", "data/snapshots")
    source = data_config.get("source", "yfinance")
    interval = data_config.get("interval", "1d")

    snapshot_dir = Path(snapshot_dir_base) / f"{source}_{interval}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Using snapshot directory: {snapshot_dir}")

    failed_symbols = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=data_config.get("start_date"),
                end=data_config.get("end_date"),
                interval=interval,
                auto_adjust=True,
                prepost=False,
                actions=False,
            )

            if data.empty:
                log.warning(f"No data returned for {symbol}. Skipping.")
                failed_symbols.append(symbol)
                continue

            parquet_path = snapshot_dir / f"{symbol}.parquet"
            data.to_parquet(parquet_path, engine="pyarrow")
            log.debug(f"Saved snapshot for {symbol} to {parquet_path}")

        except Exception as e:
            log.warning(f"Failed to fetch or save data for {symbol}: {e}")
            failed_symbols.append(symbol)
            continue

    if failed_symbols:
        log.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")


def load_snapshots(symbols: List[str], config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:  # impure
    """
    Load existing data snapshots for a list of symbols.
    """
    data_config = config.get("data", {})
    snapshot_dir_base = data_config.get("snapshot_dir", "data/snapshots")
    source = data_config.get("source", "yfinance")
    interval = data_config.get("interval", "1d")

    snapshot_dir = Path(snapshot_dir_base) / f"{source}_{interval}"
    log.info(f"Loading snapshots from: {snapshot_dir}")

    if not snapshot_dir.exists():
        raise FileNotFoundError(
            f"Snapshot directory not found: {snapshot_dir}. "
            "Run with refresh=True to fetch data."
        )

    loaded_data = {}
    for symbol in symbols:
        parquet_path = snapshot_dir / f"{symbol}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Snapshot not found for {symbol}: {parquet_path}. "
                "Run with refresh=True to fetch data."
            )
        loaded_data[symbol] = pd.read_parquet(parquet_path)

    log.info(f"Successfully loaded {len(loaded_data)} snapshots.")
    return loaded_data
