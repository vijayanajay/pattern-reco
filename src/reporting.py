"""
Generating output reports, metrics, and trade ledgers.
"""
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from src.types import Trade

__all__ = [
    "calculate_trade_returns",
    "calculate_per_stock_metrics",
    "calculate_benchmark_metrics",
    "calculate_oos_is_ratio",
]


def calculate_trade_returns(trades: List[Trade]) -> pd.DataFrame:
    """
    Calculates returns and other metrics for a list of trades.

    Args:
        trades: A list of Trade objects.

    Returns:
        A DataFrame with one row per trade, with columns for symbol,
        entry_date, exit_date, return_pct, and duration_days.
        Returns an empty DataFrame if the input list is empty.
    """
    if not trades:
        return pd.DataFrame()

    records = [t.dict() for t in trades]
    df = pd.DataFrame(records)

    # Ensure date columns are in datetime format for .dt accessor
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    df["return_pct"] = (df["exit_price"] / df["entry_price"]) - 1
    df["duration_days"] = (df["exit_date"] - df["entry_date"]).dt.days
    return df


def calculate_per_stock_metrics(trade_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates per-stock performance metrics from a trade returns DataFrame.

    Args:
        trade_returns: DataFrame from calculate_trade_returns.

    Returns:
        A DataFrame indexed by stock symbol, with columns for:
        - trade_count
        - median_return_pct
        - p5_return_pct (5th percentile)
        - hit_rate (fraction of trades with positive returns)
    """
    if trade_returns.empty:
        return pd.DataFrame()

    # Define a custom quantile function for the 5th percentile
    p5 = lambda x: x.quantile(0.05)
    p5.__name__ = "p5_return_pct"

    # Define hit rate
    hit_rate = lambda x: (x > 0).mean()
    hit_rate.__name__ = "hit_rate"

    grouped = trade_returns.groupby("symbol")["return_pct"]
    metrics = grouped.agg(["count", "median", p5, hit_rate])
    metrics.rename(columns={"count": "trade_count", "median": "median_return_pct"}, inplace=True)
    return metrics


def calculate_benchmark_metrics(benchmark_prices: pd.Series) -> Dict[str, float]:
    """
    Calculates buy-and-hold metrics for a benchmark index.

    Args:
        benchmark_prices: A Series of benchmark prices, indexed by date.

    Returns:
        A dictionary with the total return of the benchmark.
    """
    if benchmark_prices.empty:
        return {"total_return_pct": 0.0}

    total_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
    return {"total_return_pct": total_return}


def calculate_oos_is_ratio(trade_returns: pd.DataFrame) -> float:
    """
    Calculates the ratio of Out-of-Sample (OOS) to In-Sample (IS)
    performance, using median trade return as the metric.

    Args:
        trade_returns: DataFrame from calculate_trade_returns.

    Returns:
        The OOS/IS performance ratio. Returns 0.0 if there are no IS trades
        or if the IS median return is zero or negative.
    """
    if trade_returns.empty or "sample_type" not in trade_returns.columns:
        return 0.0

    median_returns = trade_returns.groupby("sample_type")["return_pct"].median()

    is_median = median_returns.get("IS")
    oos_median = median_returns.get("OOS")

    if oos_median is None:
        return 0.0  # No OOS trades, ratio is undefined/zero

    if is_median is None or is_median <= 0:
        return 0.0  # No IS trades or non-positive IS performance

    return oos_median / is_median
