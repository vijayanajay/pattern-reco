"""
Tests for the execution simulation engine.
"""
from datetime import date

import pandas as pd
import pytest

from src.config import ExecutionConfig
from src.execution import Signal, simulate_execution


@pytest.fixture
def exec_config() -> ExecutionConfig:
    """Provides a default ExecutionConfig for tests."""
    return ExecutionConfig(
        circuit_guard_pct=0.10,
        fees_bps=10.0,
        slippage_model={"gap_2pct": 5.0, "gap_5pct": 10.0, "gap_high": 20.0},
    )


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """Provides sample market data for a single stock."""
    dates = pd.to_datetime(
        [
            "2023-01-02",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
            "2023-01-06",
        ]
    )
    data = {
        "Open": [100, 102, 120, 95, 98],
        "Close": [101, 103, 122, 94, 99],
    }
    df = pd.DataFrame(data, index=dates)
    return {"TEST.NS": df}


def test_successful_buy_signal(exec_config: ExecutionConfig, sample_data: dict):
    """Tests a standard buy signal that should be filled successfully."""
    signal = Signal(symbol="TEST.NS", signal_date=date(2023, 1, 2), action="BUY", source="test")
    fills, unfilled = simulate_execution([signal], sample_data, exec_config)

    assert len(fills) == 1
    assert len(unfilled) == 0
    fill = fills[0]
    assert fill.signal == signal
    assert fill.executed_date == date(2023, 1, 3)

    # Gap = (102 - 101) / 101 = 0.99% < 2%, so slippage is 5 bps
    expected_slippage = 102 * (5.0 / 10000.0)
    expected_price = 102 + expected_slippage
    assert fill.executed_price == pytest.approx(expected_price)

    # Fees are 10 bps on the final price
    expected_fees = expected_price * (10.0 / 10000.0)
    assert fill.fees == pytest.approx(expected_fees)


def test_circuit_guard_triggered(exec_config: ExecutionConfig, sample_data: dict):
    """Tests that a trade is rejected if it exceeds the circuit guard limit."""
    # Day 3 -> Day 4: Open is 120, prev close is 103. Gap = (120-103)/103 = 16.5% > 10%
    signal = Signal(symbol="TEST.NS", signal_date=date(2023, 1, 3), action="BUY", source="test")
    fills, unfilled = simulate_execution([signal], sample_data, exec_config)

    assert len(fills) == 0
    assert len(unfilled) == 1
    fail = unfilled[0]
    assert fail.signal == signal
    assert fail.reason.startswith("Circuit guard triggered")
    assert fail.attempted_date == date(2023, 1, 4)


def test_signal_on_last_day(exec_config: ExecutionConfig, sample_data: dict):
    """Tests that a signal on the last day of data is correctly marked as unfilled."""
    signal = Signal(symbol="TEST.NS", signal_date=date(2023, 1, 6), action="BUY", source="test")
    fills, unfilled = simulate_execution([signal], sample_data, exec_config)

    assert len(fills) == 0
    assert len(unfilled) == 1
    assert unfilled[0].reason == "Signal is on the last day of available data"


def test_signal_with_no_data_for_symbol(exec_config: ExecutionConfig, sample_data: dict):
    """Tests a signal for a symbol for which no data is available."""
    signal = Signal(symbol="UNKNOWN.NS", signal_date=date(2023, 1, 2), action="BUY", source="test")
    fills, unfilled = simulate_execution([signal], sample_data, exec_config)

    assert len(fills) == 0
    assert len(unfilled) == 1
    assert unfilled[0].reason == "No data for symbol UNKNOWN.NS"


def test_signal_date_not_in_data(exec_config: ExecutionConfig, sample_data: dict):
    """Tests a signal whose date does not exist in the historical data."""
    signal = Signal(symbol="TEST.NS", signal_date=date(2023, 1, 8), action="BUY", source="test") # A Sunday
    fills, unfilled = simulate_execution([signal], sample_data, exec_config)

    assert len(fills) == 0
    assert len(unfilled) == 1
    assert unfilled[0].reason == f"Signal date {signal.signal_date} not found in data"


def test_empty_signal_list(exec_config: ExecutionConfig, sample_data: dict):
    """Tests that an empty list of signals produces no output."""
    fills, unfilled = simulate_execution([], sample_data, exec_config)
    assert len(fills) == 0
    assert len(unfilled) == 0
