"""
Universe selection logic.
"""
from datetime import date
from typing import Any, Dict, List

import pandas as pd
from rich.console import Console

from src.config import Config

__all__ = ["select_universe", "get_nse_symbols"]


def get_nse_symbols(config: Config, console: Console) -> List[str]:
    """
    Get the list of symbols to consider for the universe from the config.
    """
    if not config.universe.include_symbols:
        raise ValueError("Config must provide a list of symbols in 'universe.include_symbols'")

    console.log(f"Loaded {len(config.universe.include_symbols)} symbols from config for universe consideration.")
    return config.universe.include_symbols


def _compute_turnover(data: pd.DataFrame, lookback_days: int) -> float:
    """Computes median daily turnover for a single stock."""
    if data.empty or len(data) < lookback_days * 0.7:
        return 0.0

    turnover = data["Close"].tail(lookback_days) * data["Volume"].tail(lookback_days)
    return turnover[turnover > 0].median()


def select_universe(
    all_data: Dict[str, pd.DataFrame],
    config: Config,
    t0: date,
    console: Console,
) -> List[str]:
    """
    Selects a universe of stocks based on liquidity and price criteria at a given time t0.
    """
    cfg = config.universe
    candidate_symbols = [s for s in all_data.keys() if s not in cfg.exclude_symbols]

    console.log(f"Starting universe selection from {len(candidate_symbols)} candidates at {t0}.")

    lookback_days = int(cfg.lookback_years * 252)

    qualified_symbols = {}
    for symbol in candidate_symbols:
        data = all_data[symbol]

        # Filter data up to t0
        data_at_t0 = data[data.index.date <= t0]
        if data_at_t0.empty:
            continue

        # Apply filters
        if data_at_t0["Close"].iloc[-1] < cfg.min_price:
            continue

        median_turnover = _compute_turnover(data_at_t0, lookback_days)
        if median_turnover < cfg.min_turnover:
            continue

        qualified_symbols[symbol] = median_turnover

    if not qualified_symbols:
        raise ValueError("No stocks met the universe selection criteria.")

    # Sort by turnover and select top N
    sorted_symbols = sorted(
        qualified_symbols.items(), key=lambda item: item[1], reverse=True
    )

    selected_universe = [symbol for symbol, turnover in sorted_symbols[:cfg.size]]

    console.log(f"Selected a universe of {len(selected_universe)} symbols.")
    return selected_universe
