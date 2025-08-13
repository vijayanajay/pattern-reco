"""
Tests for universe selection logic.
"""
from datetime import date
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest
from rich.console import Console

from src.config import Config
from src.universe import select_universe


@pytest.fixture
def sample_universe_data() -> Dict[str, pd.DataFrame]:
    """Provides a sample of loaded snapshot data for universe selection tests."""
    return {
        "HIGH_TURNOVER.NS": pd.DataFrame(
            {"Close": [100.0] * 500, "Volume": [200_000] * 500},
            index=pd.to_datetime(pd.date_range("2022-01-01", periods=500)),
        ),
        "MED_TURNOVER.NS": pd.DataFrame(
            {"Close": [100.0] * 500, "Volume": [100_000] * 500},
            index=pd.to_datetime(pd.date_range("2022-01-01", periods=500)),
        ),
        "LOW_TURNOVER.NS": pd.DataFrame(
            {"Close": [100.0] * 500, "Volume": [1_000] * 500},  # Below min_turnover
            index=pd.to_datetime(pd.date_range("2022-01-01", periods=500)),
        ),
        "PENNY_STOCK.NS": pd.DataFrame(
            {"Close": [5.0] * 500, "Volume": [200_000] * 500},  # Below min_price
            index=pd.to_datetime(pd.date_range("2022-01-01", periods=500)),
        ),
        "INSUFFICIENT_DATA.NS": pd.DataFrame(
            {"Close": [100.0] * 50, "Volume": [200_000] * 50},
            index=pd.to_datetime(pd.date_range("2022-11-01", periods=50)),
        ),
    }


@pytest.fixture
def test_config() -> Config:
    """Provides a base config for universe tests."""
    return Config.parse_obj(
        {
            "run": {"name": "test"},
            "data": {"start_date": "2023-01-01", "end_date": "2023-01-31"},
            "universe": {
                "size": 2,
                "min_turnover": 1_000_000,
                "min_price": 10.0,
                "lookback_years": 1,
                "exclude_symbols": [],
                "include_symbols": [
                    "HIGH_TURNOVER.NS",
                    "MED_TURNOVER.NS",
                    "LOW_TURNOVER.NS",
                    "PENNY_STOCK.NS",
                    "INSUFFICIENT_DATA.NS",
                    "ANOTHER_OK.NS"
                ],
            },
            "detector": {}, "walk_forward": {}, "execution": {}, "portfolio": {}, "reporting": {},
        }
    )


@pytest.fixture
def console() -> Console:
    """Fixture for a Rich console that discards output."""
    return Console(file=open(Path.cwd() / "test.log", "w"))


def test_select_universe_ranking_and_filtering(
    sample_universe_data: Dict[str, pd.DataFrame], test_config: Config, console: Console
) -> None:
    """
    Tests that universe selection correctly filters by price and turnover,
    and ranks by turnover.
    """
    t0 = date(2023, 1, 1)
    selected = select_universe(sample_universe_data, test_config, t0, console)

    assert len(selected) == 2
    assert selected == ["HIGH_TURNOVER.NS", "MED_TURNOVER.NS"]
    assert "LOW_TURNOVER.NS" not in selected
    assert "PENNY_STOCK.NS" not in selected
    assert "INSUFFICIENT_DATA.NS" not in selected


def test_select_universe_exclusion_list(
    sample_universe_data: Dict[str, pd.DataFrame], test_config: Config, console: Console
) -> None:
    """Tests that the exclusion list is respected."""
    test_config.universe.exclude_symbols = ["MED_TURNOVER.NS"]
    t0 = date(2023, 1, 1)

    sample_universe_data["ANOTHER_OK.NS"] = pd.DataFrame(
        {"Close": [100.0] * 500, "Volume": [150_000] * 500},
        index=pd.to_datetime(pd.date_range("2022-01-01", periods=500)),
    )

    selected = select_universe(sample_universe_data, test_config, t0, console)

    assert "MED_TURNOVER.NS" not in selected
    assert "HIGH_TURNOVER.NS" in selected
    assert "ANOTHER_OK.NS" in selected
    assert len(selected) == 2


def test_select_universe_t0_filtering(
    sample_universe_data: Dict[str, pd.DataFrame], test_config: Config, console: Console
) -> None:
    """Tests that data is correctly filtered up to t0."""
    test_config.universe.min_turnover = 1
    test_config.universe.min_price = 1
    test_config.universe.size = 1

    t0_early = date(2022, 2, 1)
    with pytest.raises(ValueError, match="No stocks met the universe selection criteria"):
        select_universe(sample_universe_data, test_config, t0_early, console)

    t0_late = date(2023, 1, 1)
    selected_late = select_universe(sample_universe_data, test_config, t0_late, console)
    assert "HIGH_TURNOVER.NS" in selected_late


def test_select_universe_no_stocks_meet_criteria(
    sample_universe_data: Dict[str, pd.DataFrame], test_config: Config, console: Console
) -> None:
    """Tests that an error is raised if no stocks qualify."""
    test_config.universe.min_turnover = 999_999_999
    t0 = date(2023, 1, 1)

    with pytest.raises(ValueError, match="No stocks met the universe selection criteria"):
        select_universe(sample_universe_data, test_config, t0, console)
