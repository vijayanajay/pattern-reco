"""
Legacy backtesting logic.

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
]


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
