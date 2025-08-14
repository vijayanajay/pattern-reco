"""
Performance metrics and evaluation.

This module provides functions to calculate various performance metrics
for a backtest, including trade-level, stock-level, and portfolio-level
statistics.
"""
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from src.types import Trade

__all__ = [
    "calculate_trade_returns",
    "calculate_per_stock_metrics",
    "aggregate_portfolio_returns",
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


def aggregate_portfolio_returns(
    trades: List[Trade],
    daily_prices: Dict[str, pd.Series],
    initial_capital: float,
    position_size: float,
    start_date: date,
    end_date: date,
) -> pd.Series:
    """
    Creates a daily portfolio equity curve.

    Args:
        trades: List of trades.
        daily_prices: Dict mapping symbol to a Series of daily close prices.
        initial_capital: Starting capital.
        position_size: Fixed capital allocated to each trade.
        start_date: The start date of the portfolio series.
        end_date: The end date of the portfolio series.

    Returns:
        A Series of daily portfolio values, indexed by date.
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    # Initialize with float to avoid dtype warnings later
    portfolio_value = pd.Series(float(initial_capital), index=date_range)
    cash = float(initial_capital)

    # Prepare price data by forward-filling
    ffilled_prices = {}
    for symbol, prices in daily_prices.items():
        ffilled_prices[symbol] = prices.reindex(date_range, method="ffill").ffill()

    active_trades = []
    trade_idx = 0
    trades = sorted(trades, key=lambda t: t.entry_date)

    for dt in date_range:
        current_date = dt.date()
        # Exit trades
        exited_trades = []
        for trade in active_trades:
            if current_date >= trade.exit_date:
                cash += position_size * (trade.exit_price / trade.entry_price)
                exited_trades.append(trade)
        active_trades = [t for t in active_trades if t not in exited_trades]

        # Enter new trades
        while trade_idx < len(trades) and current_date >= trades[trade_idx].entry_date:
            trade = trades[trade_idx]
            if cash >= position_size:
                cash -= position_size
                active_trades.append(trade)
            trade_idx += 1

        # Value open positions
        current_holdings_value = 0.0
        for trade in active_trades:
            current_price = ffilled_prices[trade.symbol].get(dt)
            if pd.notna(current_price):
                current_holdings_value += position_size * (current_price / trade.entry_price)
            else:
                # If price is still NaN (e.g., before first data point), carry cost
                current_holdings_value += position_size

        portfolio_value.loc[dt] = cash + current_holdings_value

    return portfolio_value


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
