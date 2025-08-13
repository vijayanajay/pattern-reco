"""
Tests for universe selection logic.
"""
from datetime import date
from typing import Any, Dict

import pandas as pd
import pytest

from src.core import universe


@pytest.fixture
def sample_universe_data() -> Dict[str, pd.DataFrame]:
    """Provides a sample of loaded snapshot data for universe selection tests."""
    return {
        "HIGH_TURNOVER.NS": pd.DataFrame({
            'Close': [100.0] * 500, 'Volume': [200_000] * 500
        }, index=pd.to_datetime(pd.date_range("2022-01-01", periods=500))),
        "MED_TURNOVER.NS": pd.DataFrame({
            'Close': [100.0] * 500, 'Volume': [100_000] * 500
        }, index=pd.to_datetime(pd.date_range("2022-01-01", periods=500))),
        "LOW_TURNOVER.NS": pd.DataFrame({
            'Close': [100.0] * 500, 'Volume': [1_000] * 500 # Below min_turnover
        }, index=pd.to_datetime(pd.date_range("2022-01-01", periods=500))),
        "PENNY_STOCK.NS": pd.DataFrame({
            'Close': [5.0] * 500, 'Volume': [200_000] * 500 # Below min_price
        }, index=pd.to_datetime(pd.date_range("2022-01-01", periods=500))),
        "INSUFFICIENT_DATA.NS": pd.DataFrame({
            'Close': [100.0] * 50, 'Volume': [200_000] * 50
        }, index=pd.to_datetime(pd.date_range("2022-11-01", periods=50))),
    }

def test_get_nse_symbols() -> None:
    """Test that get_nse_symbols returns a non-empty list of strings."""
    symbols = universe.get_nse_symbols()
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert all(isinstance(s, str) for s in symbols)

def test_select_universe_ranking_and_filtering(sample_universe_data: Dict[str, pd.DataFrame]) -> None:
    """
    Tests that universe selection correctly filters by price and turnover,
    and ranks by turnover.
    """
    config: Dict[str, Any] = {
        "size": 2,
        "min_turnover": 1_000_000,
        "min_price": 10.0,
        "lookback_years": 1,
        "exclude_symbols": [],
    }
    t0 = date(2023, 1, 1)

    selected = universe.select_universe(sample_universe_data, config, t0)

    assert len(selected) == 2
    # Should be ranked by turnover
    assert selected == ["HIGH_TURNOVER.NS", "MED_TURNOVER.NS"]

    # Check that filtered stocks are not in the universe
    assert "LOW_TURNOVER.NS" not in selected
    assert "PENNY_STOCK.NS" not in selected
    assert "INSUFFICIENT_DATA.NS" not in selected

def test_select_universe_exclusion_list(sample_universe_data: Dict[str, pd.DataFrame]) -> None:
    """Tests that the exclusion list is respected."""
    config: Dict[str, Any] = {
        "size": 2,
        "min_turnover": 1_000_000,
        "min_price": 10.0,
        "lookback_years": 1,
        "exclude_symbols": ["MED_TURNOVER.NS"],
    }
    t0 = date(2023, 1, 1)

    # We need to add another valid stock to be selected
    sample_universe_data["ANOTHER_OK.NS"] = pd.DataFrame({
        'Close': [100.0] * 500, 'Volume': [150_000] * 500
    }, index=pd.to_datetime(pd.date_range("2022-01-01", periods=500)))

    selected = universe.select_universe(sample_universe_data, config, t0)

    assert "MED_TURNOVER.NS" not in selected
    assert "HIGH_TURNOVER.NS" in selected
    assert "ANOTHER_OK.NS" in selected
    assert len(selected) == 2

def test_select_universe_t0_filtering(sample_universe_data: Dict[str, pd.DataFrame]) -> None:
    """Tests that data is correctly filtered up to t0."""
    config: Dict[str, Any] = { "size": 1, "min_turnover": 1, "min_price": 1, "lookback_years": 1, "exclude_symbols": [] }

    # t0 is before most of the data for HIGH_TURNOVER, so no stock should qualify
    t0_early = date(2022, 2, 1)
    with pytest.raises(ValueError, match="No stocks met the universe selection criteria"):
        universe.select_universe(sample_universe_data, config, t0_early)

    # t0 is after all data, so HIGH_TURNOVER should be selected
    t0_late = date(2023, 1, 1)
    selected_late = universe.select_universe(sample_universe_data, config, t0_late)
    assert "HIGH_TURNOVER.NS" in selected_late

def test_select_universe_no_stocks_meet_criteria(sample_universe_data: Dict[str, pd.DataFrame]) -> None:
    """Tests that an error is raised if no stocks qualify."""
    config: Dict[str, Any] = {
        "size": 1,
        "min_turnover": 999_999_999, # Unattainable turnover
        "min_price": 10.0,
        "lookback_years": 1,
        "exclude_symbols": [],
    }
    t0 = date(2023, 1, 1)

    with pytest.raises(ValueError, match="No stocks met the universe selection criteria"):
        universe.select_universe(sample_universe_data, config, t0)
