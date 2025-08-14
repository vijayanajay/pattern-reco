"""
This module will be deprecated and replaced by a vectorbt-based implementation
in `src/backtest.py`. It is kept here temporarily to ensure the `run` command
remains functional during the refactoring.
"""
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd

from src.types import Trade

__all__ = [
    "aggregate_portfolio_returns",
    "calculate_per_stock_metrics",
    "calculate_benchmark_metrics",
    "calculate_oos_is_ratio",
]


def calculate_per_stock_metrics(trade_returns: pd.DataFrame) -> pd.DataFrame:
    if trade_returns.empty:
        return pd.DataFrame()
    p5 = lambda x: x.quantile(0.05)
    p5.__name__ = "p5_return_pct"
    hit_rate = lambda x: (x > 0).mean()
    hit_rate.__name__ = "hit_rate"
    metrics = trade_returns.groupby("symbol")["return_pct"].agg(["count", "median", p5, hit_rate])
    metrics.rename(columns={"count": "trade_count", "median": "median_return_pct"}, inplace=True)
    return metrics


def calculate_benchmark_metrics(benchmark_prices: pd.Series) -> Dict[str, float]:
    if benchmark_prices.empty:
        return {"total_return_pct": 0.0}
    total_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
    return {"total_return_pct": total_return}


def calculate_oos_is_ratio(trade_returns: pd.DataFrame) -> float:
    if trade_returns.empty or "sample_type" not in trade_returns.columns:
        return 0.0
    median_returns = trade_returns.groupby("sample_type")["return_pct"].median()
    is_median = median_returns.get("IS")
    oos_median = median_returns.get("OOS")
    if oos_median is None or is_median is None or is_median <= 0:
        return 0.0
    return oos_median / is_median


def aggregate_portfolio_returns(
    trades: List[Trade],
    daily_prices: Dict[str, pd.Series],
    initial_capital: float,
    position_size: float,
    start_date: date,
    end_date: date,
) -> pd.Series:
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    portfolio_value = pd.Series(float(initial_capital), index=date_range)
    cash = float(initial_capital)
    ffilled_prices = {s: p.reindex(date_range, method="ffill").ffill() for s, p in daily_prices.items()}
    active_trades: List[Trade] = []
    trade_idx = 0
    trades = sorted(trades, key=lambda t: t.entry_date)
    for dt in date_range:
        cash += sum(
            position_size * (t.exit_price / t.entry_price) for t in active_trades if dt.date() >= t.exit_date
        )
        active_trades = [t for t in active_trades if dt.date() < t.exit_date]
        while trade_idx < len(trades) and dt.date() >= trades[trade_idx].entry_date:
            if cash >= position_size:
                cash -= position_size
                active_trades.append(trades[trade_idx])
            trade_idx += 1
        current_holdings_value = sum(
            position_size * (ffilled_prices[t.symbol].get(dt, t.entry_price) / t.entry_price)
            for t in active_trades
        )
        portfolio_value.loc[dt] = cash + current_holdings_value
    return portfolio_value
