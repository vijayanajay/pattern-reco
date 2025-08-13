"""
Universe selection logic.
"""
import pandas as pd
from rich.console import Console

from src.config import Config
from src.data import load_snapshots, discover_symbols

__all__ = ["select_universe"]

TRADING_DAYS_PER_YEAR = 252


def select_universe(config: Config, console: Console) -> list[str]:
    """
    Selects a universe of stocks based on liquidity and price criteria.

    This function discovers all available symbols from the snapshot directory,
    loads their data, and then filters and ranks them based on the criteria
    defined in the configuration at the specified time `t0`.
    """
    cfg = config.universe
    t0 = config.run.t0
    console.log(f"Starting universe selection from all available symbols at {t0}...")

    all_symbols = discover_symbols(config)
    if not all_symbols:
        raise ValueError("No data snapshots found. Cannot select universe.")

    candidate_symbols = [s for s in all_symbols if s not in cfg.exclude_symbols]
    all_data = load_snapshots(candidate_symbols, config, console)

    if not all_data:
        raise ValueError("No data loaded for any candidate symbols.")

    lookback_days = int(cfg.lookback_years * TRADING_DAYS_PER_YEAR)

    df_list = []
    for symbol, df in all_data.items():
        df["symbol"] = symbol
        df_list.append(df)

    full_df = pd.concat(df_list)
    full_df = full_df[full_df.index.date <= t0]

    # 1. Filter by latest price
    latest_prices = full_df.groupby("symbol")["Close"].last()
    price_filtered_symbols = latest_prices[latest_prices >= cfg.min_price].index

    df_filtered = full_df[full_df["symbol"].isin(price_filtered_symbols)].copy()

    # 2. Calculate median turnover
    df_filtered["turnover"] = df_filtered["Close"] * df_filtered["Volume"]

    # Ensure we have enough data for turnover calculation
    lookback_df = df_filtered.groupby("symbol").tail(lookback_days)

    median_turnover = lookback_df.groupby("symbol")["turnover"].median()

    # 3. Filter by turnover
    turnover_filtered_symbols = median_turnover[median_turnover >= cfg.min_turnover].index

    # 4. Final ranking and selection
    final_ranks = median_turnover[turnover_filtered_symbols].sort_values(ascending=False)

    selected_universe = list(final_ranks.head(cfg.size).index)

    if not selected_universe:
        raise ValueError("No stocks met the universe selection criteria.")

    console.log(f"Selected a universe of {len(selected_universe)} symbols.")
    return selected_universe
