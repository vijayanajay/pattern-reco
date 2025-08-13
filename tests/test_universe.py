"""
Tests for universe selection logic.
"""
from pathlib import Path

import pandas as pd
import pytest
from rich.console import Console

from src.config import Config
from src.universe import select_universe


@pytest.fixture
def snapshot_dir(tmp_path: Path) -> Path:
    """Creates a temporary snapshot directory with fake data."""
    source = "yfinance_test"
    interval = "1d"
    snapshot_path = tmp_path / f"{source}_{interval}"
    snapshot_path.mkdir()

    # High turnover, valid
    pd.DataFrame(
        {"Close": [100.0] * 260, "Volume": [200_000] * 260},
        index=pd.to_datetime(pd.date_range("2022-01-01", periods=260)),
    ).to_parquet(snapshot_path / "HIGH_TURNOVER.NS.parquet")

    # Medium turnover, valid
    pd.DataFrame(
        {"Close": [100.0] * 260, "Volume": [100_000] * 260},
        index=pd.to_datetime(pd.date_range("2022-01-01", periods=260)),
    ).to_parquet(snapshot_path / "MED_TURNOVER.NS.parquet")

    # Low turnover, should be filtered out
    pd.DataFrame(
        {"Close": [100.0] * 260, "Volume": [1_000] * 260},
        index=pd.to_datetime(pd.date_range("2022-01-01", periods=260)),
    ).to_parquet(snapshot_path / "LOW_TURNOVER.NS.parquet")

    # Penny stock, should be filtered out
    pd.DataFrame(
        {"Close": [5.0] * 260, "Volume": [200_000] * 260},
        index=pd.to_datetime(pd.date_range("2022-01-01", periods=260)),
    ).to_parquet(snapshot_path / "PENNY_STOCK.NS.parquet")

    return tmp_path


@pytest.fixture
def test_config(snapshot_dir: Path) -> Config:
    """Provides a base config for universe tests pointing to the temp snapshot dir."""
    return Config.parse_obj(
        {
            "run": {"name": "test", "t0": "2023-01-01"},
            "data": {
                "source": "yfinance_test",
                "interval": "1d",
                "start_date": "2022-01-01",
                "end_date": "2023-01-31",
                "snapshot_dir": snapshot_dir,
            },
            "universe": {
                "size": 2,
                "min_turnover": 10_000_000,
                "min_price": 10.0,
                "lookback_years": 1,
            },
            "detector": {}, "walk_forward": {}, "execution": {},
            "portfolio": {}, "reporting": {},
        }
    )


@pytest.fixture
def console() -> Console:
    """Fixture for a Rich console that discards output."""
    return Console(file=open("test.log", "w"))


def test_select_universe_ranking_and_filtering(
    test_config: Config, console: Console
) -> None:
    """
    Tests that universe selection correctly discovers, filters, and ranks symbols.
    """
    selected = select_universe(test_config, console)

    assert len(selected) == 2
    # Should be sorted by turnover descending
    assert selected == ["HIGH_TURNOVER.NS", "MED_TURNOVER.NS"]


def test_select_universe_exclusion_list(
    test_config: Config, console: Console
) -> None:
    """Tests that the exclusion list is respected."""
    test_config.universe.exclude_symbols = ["HIGH_TURNOVER.NS"]
    selected = select_universe(test_config, console)
    assert "HIGH_TURNOVER.NS" not in selected
    assert "MED_TURNOVER.NS" in selected


def test_select_universe_no_stocks_meet_criteria(
    test_config: Config, console: Console
) -> None:
    """Tests that an error is raised if no stocks qualify."""
    test_config.universe.min_turnover = 999_999_999
    with pytest.raises(ValueError, match="No stocks met the universe selection criteria"):
        select_universe(test_config, console)


def test_select_universe_no_snapshots_found(
    test_config: Config, console: Console
) -> None:
    """Tests that an error is raised if the snapshot directory is empty."""
    test_config.data.snapshot_dir = test_config.data.snapshot_dir / "nonexistent"
    with pytest.raises(ValueError, match="No data snapshots found"):
        select_universe(test_config, console)
