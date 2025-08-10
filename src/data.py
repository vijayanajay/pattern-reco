"""
Data fetching and snapshot management.
Simple yfinance integration without premature abstractions.
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

__all__ = ["fetch_and_snapshot", "load_snapshots"]


def fetch_and_snapshot(symbols: List[str], config: Dict[str, Any]) -> None:  # impure
    """
    Fetch data from yfinance and save to parquet snapshots.
    
    Args:
        symbols: List of stock symbols to fetch
        config: Configuration dictionary with data parameters
        
    Raises:
        RuntimeError: If data fetching fails
    """
    data_config = config.get("data", {})
    start_date = str(data_config.get("start_date"))
    end_date = str(data_config.get("end_date"))
    interval = data_config.get("interval", "1d")
    
    # Create snapshots directory
    snapshot_dir = Path("data/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    failed_symbols = []
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,
                prepost=False,
                actions=False
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Save to parquet
            parquet_path = snapshot_dir / f"{symbol}.parquet"
            data.to_parquet(parquet_path, engine='pyarrow')
            
        except (ValueError, RuntimeError) as e:
            failed_symbols.append((symbol, str(e)))
            continue
    
    if failed_symbols:
        error_msg = f"Failed to fetch {len(failed_symbols)} symbols: {failed_symbols}"
        raise RuntimeError(error_msg)


def load_snapshots(symbols: List[str]) -> Dict[str, pd.DataFrame]:  # impure
    """
    Load existing snapshots.
    
    Args:
        symbols: List of symbols to load
        
    Returns:
        Dictionary mapping symbol to DataFrame
        
    Raises:
        FileNotFoundError: If snapshots are missing
    """
    snapshot_dir = Path("data/snapshots")
    
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
    
    return loaded_data