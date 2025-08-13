"""
Tests for the backtesting engine.
"""
from typing import Any, Dict

import pandas as pd
import pytest

from src.core.backtest import run_backtest


@pytest.fixture
def sample_trade_data() -> pd.DataFrame:
    """Provides a standard OHLC DataFrame for trade simulation tests."""
    return pd.DataFrame({
        "Open":  [100, 102, 101, 103, 105, 110, 112, 115, 118, 120],
        "Close": [102, 101, 103, 105, 106, 111, 114, 117, 119, 122],
    }, index=pd.to_datetime(pd.date_range("2023-01-01", periods=10)))

@pytest.fixture
def sample_backtest_config() -> Dict[str, Any]:
    """Provides a sample config for backtesting."""
    return {
        "portfolio": {"max_hold_days": 2},
        "execution": {
            "fees_bps": 10,
            "circuit_guard_pct": 0.10,
            "slippage_model": {"gap_2pct": 5, "gap_5pct": 10, "gap_high": 20},
        },
    }

def test_run_backtest_normal_case(
    sample_trade_data: pd.DataFrame, sample_backtest_config: Dict[str, Any]
) -> None:
    """Tests a typical scenario with a successful trade."""
    signals = pd.Series([False, True] + [False] * 8, index=sample_trade_data.index)
    trade_ledger = run_backtest(sample_trade_data, signals, sample_backtest_config)

    assert len(trade_ledger) == 1
    trade = trade_ledger.iloc[0]

    # Entry: signal @ 1 -> enter @ 2 (Open = 101)
    # Exit: hold 2 days -> exit @ 4 (Close = 105)
    assert trade["entry_date"] == sample_trade_data.index[2]
    assert trade["exit_date"] == sample_trade_data.index[4]
    assert trade["entry_price"] == 101

    # Costs: gap = (101 - 101)/101 = 0 -> 5 bps slippage
    # entry_adj = 101 * (1 + 5/10000) = 101.0505
    # return = (106 - 101.0505) / 101.0505 = 0.04898...
    assert pytest.approx(trade["return_pct"], 1e-4) == 0.04898
    assert trade["slippage_bps"] == 5
    assert trade["fees_bps"] == 10

def test_run_backtest_circuit_breaker(
    sample_trade_data: pd.DataFrame, sample_backtest_config: Dict[str, Any]
) -> None:
    """Tests that the circuit breaker prevents a trade."""
    # Modify data to have a large gap
    data = sample_trade_data.copy()
    data.loc[data.index[2], "Open"] = 120 # Prev close is 101, so > 10% gap

    signals = pd.Series([False, True] + [False] * 8, index=data.index)
    trade_ledger = run_backtest(data, signals, sample_backtest_config)

    assert trade_ledger.empty

def test_run_backtest_no_signals(
    sample_trade_data: pd.DataFrame, sample_backtest_config: Dict[str, Any]
) -> None:
    """Ensures no trades are generated when there are no signals."""
    signals = pd.Series([False] * len(sample_trade_data), index=sample_trade_data.index)
    trade_ledger = run_backtest(sample_trade_data, signals, sample_backtest_config)
    assert trade_ledger.empty

def test_run_backtest_out_of_bounds(
    sample_trade_data: pd.DataFrame, sample_backtest_config: Dict[str, Any]
) -> None:
    """Tests that signals too close to the end of the data are ignored."""
    signals = pd.Series([False] * 8 + [True, False], index=sample_trade_data.index)
    trade_ledger = run_backtest(sample_trade_data, signals, sample_backtest_config)
    assert trade_ledger.empty
