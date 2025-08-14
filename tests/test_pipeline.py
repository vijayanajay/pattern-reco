"""
Tests for the core pipeline logic.
"""
from typing import Dict, Set

import pandas as pd
import pytest

# Correctly import the internal function for testing
from src.pipeline import Position, _select_positions


@pytest.fixture
def sample_data() -> Dict[str, pd.DataFrame]:
    """Fixture for sample historical data for multiple stocks."""
    t0 = pd.Timestamp("2023-01-05")
    return {
        "STOCK_A": pd.DataFrame(
            {"Open": [100], "Close": [100], "Volume": [1000], "Turnover": [100_000]},
            index=[t0],
        ),
        "STOCK_B": pd.DataFrame(
            {"Open": [200], "Close": [200], "Volume": [2000], "Turnover": [400_000]},
            index=[t0],
        ),
        "STOCK_C": pd.DataFrame(
            {"Open": [300], "Close": [300], "Volume": [500], "Turnover": [150_000]},
            index=[t0],
        ),
    }


@pytest.fixture
def sample_portfolio_cfg() -> Dict:
    """Fixture for a sample portfolio configuration."""
    return {
        "max_concurrent": 2,
        "position_size": 100_000,
        "reentry_lockout": True,
    }


def test_select_positions_basic(sample_data, sample_portfolio_cfg):
    """Test basic position selection based on signal strength."""
    t0 = pd.Timestamp("2023-01-05")
    signals = pd.DataFrame([
        {"date": t0, "symbol": "STOCK_A", "signal": 0.9},
        {"date": t0, "symbol": "STOCK_B", "signal": 0.7},
        {"date": t0, "symbol": "STOCK_C", "signal": 0.8},
    ])

    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, set())

    assert len(new_positions) == 2
    assert new_positions[0].symbol == "STOCK_A" # Highest signal
    assert new_positions[1].symbol == "STOCK_C" # Second highest signal


def test_select_positions_tiebreak_turnover(sample_data, sample_portfolio_cfg):
    """Test turnover tie-breaking when signals are equal."""
    t0 = pd.Timestamp("2023-01-05")
    signals = pd.DataFrame([
        {"date": t0, "symbol": "STOCK_A", "signal": 0.9}, # Lower turnover
        {"date": t0, "symbol": "STOCK_B", "signal": 0.9}, # Higher turnover
        {"date": t0, "symbol": "STOCK_C", "signal": 0.7},
    ])

    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, set())

    assert len(new_positions) == 2
    assert new_positions[0].symbol == "STOCK_B" # Higher turnover wins
    assert new_positions[1].symbol == "STOCK_A"


def test_select_positions_tiebreak_symbol(sample_data, sample_portfolio_cfg):
    """Test symbol tie-breaking when signal and turnover are equal."""
    t0 = pd.Timestamp("2023-01-05")
    # Give B and C the same high turnover
    sample_data["STOCK_C"]["Turnover"] = 400_000
    signals = pd.DataFrame([
        {"date": t0, "symbol": "STOCK_C", "signal": 0.9}, # Alphabetically second
        {"date": t0, "symbol": "STOCK_B", "signal": 0.9}, # Alphabetically first
    ])

    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, set())

    assert len(new_positions) == 2
    assert new_positions[0].symbol == "STOCK_B" # Alphabetical order wins
    assert new_positions[1].symbol == "STOCK_C"


def test_select_positions_capacity_limit(sample_data, sample_portfolio_cfg):
    """Test that selection respects max_concurrent positions."""
    sample_portfolio_cfg["max_concurrent"] = 1
    t0 = pd.Timestamp("2023-01-05")
    signals = pd.DataFrame([
        {"date": t0, "symbol": "STOCK_A", "signal": 0.9},
        {"date": t0, "symbol": "STOCK_B", "signal": 0.8},
    ])

    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, set())
    assert len(new_positions) == 1
    assert new_positions[0].symbol == "STOCK_A"


def test_select_positions_reentry_lockout(sample_data, sample_portfolio_cfg):
    """Test that existing open positions are not re-entered."""
    t0 = pd.Timestamp("2023-01-05")
    signals = pd.DataFrame([
        {"date": t0, "symbol": "STOCK_A", "signal": 0.9}, # Already open
        {"date": t0, "symbol": "STOCK_B", "signal": 0.8},
        {"date": t0, "symbol": "STOCK_C", "signal": 0.7},
    ])
    open_positions = {"STOCK_A"}

    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, open_positions)

    assert len(new_positions) == 1 # Only one slot left, A is locked out
    assert new_positions[0].symbol == "STOCK_B" # B is selected as it's next best


def test_select_positions_no_slots(sample_data, sample_portfolio_cfg):
    """Test that no positions are selected if portfolio is full."""
    t0 = pd.Timestamp("2023-01-05")
    signals = pd.DataFrame([{"date": t0, "symbol": "STOCK_C", "signal": 0.9}])
    open_positions = {"STOCK_A", "STOCK_B"} # Portfolio is full

    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, open_positions)
    assert len(new_positions) == 0

def test_select_positions_no_signals(sample_data, sample_portfolio_cfg):
    """Test that no positions are selected if there are no signals."""
    signals = pd.DataFrame([])
    new_positions = _select_positions(signals, sample_data, sample_portfolio_cfg, set())
    assert len(new_positions) == 0
