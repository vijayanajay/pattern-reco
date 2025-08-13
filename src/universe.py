"""
Universe selection logic.
"""
from datetime import timedelta

import pandas as pd
from rich.console import Console

from src.config import Config
from src.data import discover_symbols, load_snapshots

__all__ = ["select_universe"]


def _get_symbol_metrics(symbol: str, config: Config, console: Console) -> dict | None:
    """Calculate filtering metrics for a single symbol."""
    cfg = config.universe
    t0 = config.run.t0

    data = load_snapshots([symbol], config, console)
    if not data:
        return None

    df = data[symbol]
    df_at_t0 = df[df.index.date <= t0]

    if df_at_t0.empty:
        return None

    latest_price = df_at_t0["Close"].iloc[-1]
    if latest_price < cfg.min_price:
        return None

    lookback_start = t0 - timedelta(days=int(cfg.lookback_years * 365.25))
    lookback_df = df_at_t0[df_at_t0.index.date > lookback_start]

    turnover = (lookback_df["Close"] * lookback_df["Volume"]).median()
    if turnover < cfg.min_turnover:
        return None

    return {"symbol": symbol, "turnover": turnover}


def select_universe(config: Config, console: Console) -> list[str]:
    """
    Selects a universe of stocks based on liquidity and price criteria.
    Processes one symbol at a time to conserve memory.
    """
    cfg = config.universe
    t0 = config.run.t0
    console.log(f"Starting universe selection from all available symbols at {t0}...")

    all_symbols = discover_symbols(config)
    if not all_symbols:
        raise ValueError("No data snapshots found. Cannot select universe.")

    candidate_symbols = [s for s in all_symbols if s not in cfg.exclude_symbols]

    metrics = []
    for symbol in candidate_symbols:
        metric = _get_symbol_metrics(symbol, config, console)
        if metric:
            metrics.append(metric)

    if not metrics:
        raise ValueError("No stocks met the universe selection criteria.")

    # Sort by turnover descending and select top N
    sorted_metrics = sorted(metrics, key=lambda x: x["turnover"], reverse=True)
    selected_universe = [m["symbol"] for m in sorted_metrics[: cfg.size]]

    console.log(f"Selected a universe of {len(selected_universe)} symbols.")
    return selected_universe
