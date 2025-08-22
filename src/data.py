"""
Data fetching, universe selection, and snapshot management.
"""
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf
from rich.console import Console

from src.config import Config

__all__ = ["fetch_and_snapshot", "load_snapshots", "discover_symbols", "select_universe"]


def _get_run_metadata(config: Config) -> Dict[str, str]:
    """Generates metadata for the data snapshot."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).strip().decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "unknown"
    return {
        "fetch_utc": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": yf.__version__,
        "git_hash": git_hash,
        "run_name": config.run.name,
    }


def _get_snapshot_dir(config: Config) -> Path:
    """Constructs the snapshot directory path from config."""
    return config.data.snapshot_dir / f"{config.data.source}_{config.data.interval}"


def discover_symbols(config: Config) -> List[str]:
    """Discovers all available symbols by scanning the snapshot directory."""
    snapshot_dir = _get_snapshot_dir(config)
    if not snapshot_dir.exists():
        return []
    return sorted([p.stem for p in snapshot_dir.glob("*.parquet")])


# impure
def select_universe(config: Config, console: Console) -> List[str]:
    """
    Selects the top N symbols based on median turnover and other criteria.
    #impure: Reads from the filesystem.
    """
    console.print("Selecting universe...")
    all_symbols = discover_symbols(config)

    # If no snapshots exist, fall back to the explicit list in config.
    if not all_symbols:
        console.print("[yellow]No snapshots found. Using 'include_symbols' from config.[/yellow]")
        return config.universe.include_symbols

    t0 = config.run.t0
    lookback_start = t0 - pd.DateOffset(years=config.universe.lookback_years)
    snapshot_dir = _get_snapshot_dir(config)

    turnover_data = []
    console.print(f"Screening {len(all_symbols)} symbols for universe selection...")
    for symbol in all_symbols:
        try:
            df = pd.read_parquet(snapshot_dir / f"{symbol}.parquet")

            # Filter for the lookback period before the run's start time (t0)
            # Convert timestamp to date for comparison to avoid TypeError
            lookback_start_date = lookback_start.date()
            df_lookback = df[(df.index.date >= lookback_start_date) & (df.index.date < t0)]

            if df_lookback.empty:
                continue

            # Price and turnover filters
            last_close = df_lookback["Close"].iloc[-1]
            if last_close < config.universe.min_price:
                continue

            # Turnover = Close * Volume
            df_lookback = df_lookback.assign(Turnover=df_lookback["Close"] * df_lookback["Volume"])
            median_turnover = df_lookback["Turnover"].median()

            if median_turnover < config.universe.min_turnover:
                continue

            turnover_data.append({"symbol": symbol, "median_turnover": median_turnover})

        except (FileNotFoundError, KeyError, IndexError):
            # Ignore symbols if data is missing, malformed, or has no rows in lookback.
            continue

    if not turnover_data:
        console.print("[bold red]Error: No symbols passed the universe selection criteria.[/bold red]")
        return []

    # Rank by median turnover and select the top N symbols
    ranked_symbols = sorted(turnover_data, key=lambda x: x["median_turnover"], reverse=True)

    selected_symbols = [d["symbol"] for d in ranked_symbols[:config.universe.size]]

    # Apply manual exclusions
    final_universe = [s for s in selected_symbols if s not in config.universe.exclude_symbols]

    console.print(f"Selected {len(final_universe)} symbols for the universe.")
    return final_universe


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
            data = yf.download(
                tickers=symbol,
                start=config.data.start_date,
                end=config.data.end_date,
                interval=config.data.interval,
                auto_adjust=True,
                prepost=False,
                actions=False,
                progress=False,
            )
            if data.empty:
                raise ValueError(f"No data returned for symbol {symbol}")

            table = pa.Table.from_pandas(data)
            metadata = _get_run_metadata(config)
            table = table.replace_schema_metadata({
                **table.schema.metadata,
                **{k.encode(): str(v).encode() for k, v in metadata.items()}
            })
            pq.write_table(table, snapshot_dir / f"{symbol}.parquet")

        except Exception:
            failed_symbols.append(symbol)

    return failed_symbols


# impure
def load_snapshots(
    symbols: List[str], config: Config, console: Console
) -> Dict[str, pd.DataFrame]:
    """
    Load existing data snapshots for a list of symbols.
    #impure: Reads from the filesystem.
    """
    snapshot_dir = _get_snapshot_dir(config)
    if not snapshot_dir.exists():
        console.print(f"[bold red]Snapshot directory not found: {snapshot_dir}[/bold red]")
        raise FileNotFoundError(f"Snapshot directory not found: {snapshot_dir}")

    loaded_data = {}
    for symbol in symbols:
        parquet_path = snapshot_dir / f"{symbol}.parquet"
        if not parquet_path.is_file():
            console.print(f"[bold red]Missing snapshot for symbol: {symbol} at {parquet_path}[/bold red]")
            raise FileNotFoundError(f"Missing snapshot for symbol: {symbol} at {parquet_path}")

        df = pd.read_parquet(parquet_path)
        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Data for {symbol} is missing required columns.")

        loaded_data[symbol] = df

    return loaded_data
