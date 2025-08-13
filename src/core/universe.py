"""
Universe selection logic.
"""
import logging
from datetime import date
from typing import Any, Dict, List

import pandas as pd

__all__ = ["select_universe", "get_nse_symbols"]

log = logging.getLogger(__name__)


def get_nse_symbols() -> List[str]:
    """
    Get a static list of NSE symbols for universe selection.
    # TODO: Externalize this list to a configurable file.
    """
    large_cap = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS",
    ]
    small_cap = [
        "ADANIPORTS.NS", "ADANIENT.NS", "GODREJCP.NS", "DIVISLAB.NS", "DRREDDY.NS",
        "EICHERMOT.NS", "GRASIM.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "JINDALSTEL.NS",
    ]
    symbols = sorted(list(set(large_cap + small_cap)))
    log.info(f"Loaded {len(symbols)} static symbols for universe consideration.")
    return symbols


def _compute_turnover(data: pd.DataFrame, lookback_days: int) -> float:
    """Computes median daily turnover for a single stock."""
    if data.empty or len(data) < lookback_days * 0.7:
        return 0.0

    turnover = data["Close"].tail(lookback_days) * data["Volume"].tail(lookback_days)
    return turnover[turnover > 0].median()


def select_universe(
    all_data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    t0: date,
) -> List[str]:
    """
    Selects a universe of stocks based on liquidity and price criteria at a given time t0.
    """

    exclude_symbols = config.get("exclude_symbols", [])
    candidate_symbols = [s for s in all_data.keys() if s not in exclude_symbols]

    log.info(f"Starting universe selection from {len(candidate_symbols)} candidates at {t0}.")

    lookback_years = config.get("lookback_years", 2)
    min_price = config.get("min_price", 10.0)
    min_turnover = config.get("min_turnover", 10_000_000.0)
    size = config.get("size", 10)

    lookback_days = int(lookback_years * 252)

    qualified_symbols = {}
    for symbol in candidate_symbols:
        data = all_data[symbol]

        # Filter data up to t0
        data_at_t0 = data[data.index.date <= t0]
        if data_at_t0.empty:
            continue

        # Apply filters
        if data_at_t0["Close"].iloc[-1] < min_price:
            continue

        median_turnover = _compute_turnover(data_at_t0, lookback_days)
        if median_turnover < min_turnover:
            continue

        qualified_symbols[symbol] = median_turnover

    if not qualified_symbols:
        raise ValueError("No stocks met the universe selection criteria.")

    # Sort by turnover and select top N
    sorted_symbols = sorted(
        qualified_symbols.items(), key=lambda item: item[1], reverse=True
    )

    selected_universe = [symbol for symbol, turnover in sorted_symbols[:size]]

    log.info(f"Selected a universe of {len(selected_universe)} symbols.")
    return selected_universe
