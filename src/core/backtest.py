"""
A simple, realistic backtesting engine.
"""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

__all__ = ["run_backtest"]

log = logging.getLogger(__name__)


def _calculate_slippage_bps(gap_pct: float, slippage_model: Dict[str, float]) -> float:
    """Calculates slippage in basis points based on the price gap."""
    gap_abs = abs(gap_pct)
    if gap_abs < 0.02:
        return slippage_model.get("gap_2pct", 5.0)
    if gap_abs < 0.05:
        return slippage_model.get("gap_5pct", 10.0)
    return slippage_model.get("gap_high", 20.0)


def run_backtest(
    data: pd.DataFrame,
    signals: pd.Series,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Runs a backtest with realistic execution assumptions.

    Args:
        data: DataFrame with OHLCV data.
        signals: Boolean series indicating trade entry signals.
        config: Dictionary with execution and portfolio parameters.

    Returns:
        A DataFrame containing the trade ledger.
    """
    exec_config = config.get("execution", {})
    portfolio_config = config.get("portfolio", {})

    max_hold = portfolio_config.get("max_hold_days", 22)
    fees_bps = exec_config.get("fees_bps", 10.0)
    circuit_guard_pct = exec_config.get("circuit_guard_pct", 0.10)
    slippage_model = exec_config.get("slippage_model", {})

    trades = []
    unfilled_trades = []

    signal_indices = np.where(signals)[0]

    for signal_idx in signal_indices:
        entry_idx = signal_idx + 1
        exit_idx = entry_idx + max_hold

        if exit_idx >= len(data):
            continue  # Not enough data to complete the trade

        entry_date = data.index[entry_idx]
        prev_close = data["Close"].iloc[signal_idx]
        entry_price = data["Open"].iloc[entry_idx]

        # 1. Circuit Breaker
        if abs(entry_price / prev_close - 1) > circuit_guard_pct:
            unfilled_trades.append({"date": entry_date, "reason": "circuit_breaker"})
            continue

        # 2. Calculate Slippage
        gap_pct = (entry_price - prev_close) / prev_close
        slippage_bps = _calculate_slippage_bps(gap_pct, slippage_model)

        # Apply slippage (add to entry price for long positions)
        entry_price_adj = entry_price * (1 + slippage_bps / 10000.0)

        # 3. Calculate Fees
        exit_price = data["Close"].iloc[exit_idx]
        entry_fee = entry_price_adj * (fees_bps / 10000.0)
        exit_fee = exit_price * (fees_bps / 10000.0)

        # 4. Record Trade
        trade = {
            "entry_date": entry_date,
            "exit_date": data.index[exit_idx],
            "hold_days": max_hold,
            "entry_price": entry_price,
            "entry_price_adj": entry_price_adj,
            "exit_price": exit_price,
            "return_pct": (exit_price - entry_price_adj) / entry_price_adj,
            "fees_bps": fees_bps,
            "slippage_bps": slippage_bps,
        }
        trades.append(trade)

    log.info(f"Backtest complete. Executed {len(trades)} trades, {len(unfilled_trades)} unfilled.")

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)
