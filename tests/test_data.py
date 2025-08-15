"""Tests for data fetching, snapshot management, and universe selection."""
import copy
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import date
from typing import Dict, Any, cast
from pytest_mock import MockerFixture

import pandas as pd
import pytest
from rich.console import Console

from src.config import Config, _from_dict
from src.data import (
    fetch_and_snapshot,
    load_snapshots,
    select_universe,
    refresh_market_data,
)

# A complete and valid dictionary for creating a Config object in tests.
FULL_CONFIG_DICT: Dict[str, Any] = {
    "run": {"name": "test_run", "t0": date(2023, 1, 15), "seed": 42, "output_dir": ""},
    "data": {
        "start_date": date(2023, 1, 1), "end_date": date(2023, 1, 31),
        "source": "yfinance_test", "interval": "1d", "snapshot_dir": "", "refresh": False,
    },
    "universe": {
        "include_symbols": ["TEST.NS", "INVALID.NS", "MISSING.NS"], "exclude_symbols": [],
        "size": 10, "min_turnover": 1e6, "min_price": 10.0, "lookback_years": 1,
    },
    "detector": {"name": "gap_z", "window_range": [20], "k_low_range": [-2.0], "max_hold": 10, "min_hit_rate": 0.0},
    "walk_forward": {"is_years": 1, "oos_years": 1, "holdout_years": 1},
    "execution": {"circuit_guard_pct": 0.1, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 1, "gap_5pct": 2, "gap_high": 3}},
    "portfolio": {"max_concurrent": 1, "position_size": 1.0, "equal_weight": True, "reentry_lockout": True},
    "reporting": {"generate_plots": False, "output_formats": ["json"], "include_unfilled": True},
}


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Pytest fixture to create a valid Config dataclass object for testing."""
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["data"]["snapshot_dir"] = str(tmp_path)
    config_dict["run"]["output_dir"] = str(tmp_path)
    # Cast is used here because _from_dict is too dynamic for mypy
    return cast(Config, _from_dict(Config, config_dict))


@patch("src.data.yf.download")
def test_fetch_and_snapshot_success(mock_download: Mock, test_config: Config) -> None:
    """Test that successful data fetching saves a snapshot."""
    mock_data = pd.DataFrame({"Open": [99.0], "High": [101.0], "Low": [98.0], "Close": [100.0], "Volume": [1000.0]})
    mock_download.return_value = mock_data

    failed = fetch_and_snapshot(["TEST.NS"], test_config)
    assert not failed
    snapshot_dir = Path(test_config.data.snapshot_dir)
    expected_path = snapshot_dir / f"{test_config.data.source}_{test_config.data.interval}" / "TEST.NS.parquet"
    assert expected_path.exists()


@patch("src.data.yf.download")
def test_fetch_and_snapshot_failure(mock_download: Mock, test_config: Config) -> None:
    """Test that yfinance failures are caught and returned."""
    mock_download.side_effect = Exception("yfinance error")
    failed = fetch_and_snapshot(["FAIL.NS"], test_config)
    assert failed == ["FAIL.NS"]


def test_load_snapshots_success(test_config: Config) -> None:
    """Test that a snapshot can be loaded successfully."""
    snapshot_dir = Path(test_config.data.snapshot_dir)
    snapshot_subdir = snapshot_dir / f"{test_config.data.source}_{test_config.data.interval}"
    snapshot_subdir.mkdir()
    fake_snapshot_path = snapshot_subdir / "TEST.NS.parquet"
    pd.DataFrame({"Open": [99], "High": [101], "Low": [98], "Close": [100.0], "Volume": [1000.0]}).to_parquet(fake_snapshot_path)
    data = load_snapshots(["TEST.NS"], test_config, Console())
    assert "TEST.NS" in data


def test_load_snapshots_missing_dir_raises_error(test_config: Config) -> None:
    """Test that loading from a non-existent snapshot directory raises an error."""
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["data"]["snapshot_dir"] = str(test_config.data.snapshot_dir / "non_existent")
    bad_config = cast(Config, _from_dict(Config, config_dict))
    with pytest.raises(FileNotFoundError):
        load_snapshots(["TEST.NS"], bad_config, Console())


def test_select_universe(test_config: Config, tmp_path: Path) -> None:
    """Test the universe selection logic."""
    snapshot_dir = Path(test_config.data.snapshot_dir) / f"{test_config.data.source}_{test_config.data.interval}"
    snapshot_dir.mkdir(exist_ok=True)

    # Stock A: High turnover, meets criteria
    df_a = pd.DataFrame({"Close": [100, 101], "Volume": [20000, 20000]}, index=pd.to_datetime(["2023-01-10", "2023-01-11"]))
    df_a.to_parquet(snapshot_dir / "STOCK_A.NS.parquet")

    # Stock B: Low price, should be excluded
    df_b = pd.DataFrame({"Close": [5, 5], "Volume": [20000, 20000]}, index=pd.to_datetime(["2023-01-10", "2023-01-11"]))
    df_b.to_parquet(snapshot_dir / "STOCK_B.NS.parquet")

    # Modify config for this specific test case
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["data"]["snapshot_dir"] = str(tmp_path)
    config_dict["universe"]["size"] = 1
    modified_config = cast(Config, _from_dict(Config, config_dict))

    universe = select_universe(modified_config, Console())
    assert universe == ["STOCK_A.NS"]


def test_refresh_market_data_existing(mocker: MockerFixture, test_config: Config) -> None:
    """Tests refresh when symbols are discovered on disk."""
    m_discover = mocker.patch("src.data.discover_symbols", return_value=["EXISTING.NS"])
    m_fetch = mocker.patch("src.data.fetch_and_snapshot", return_value=[])

    refresh_market_data(test_config, Console())

    m_discover.assert_called_once_with(test_config)
    m_fetch.assert_called_once_with(["EXISTING.NS"], test_config)


def test_refresh_market_data_fallback_to_config(mocker: MockerFixture, test_config: Config) -> None:
    """Tests refresh when no snapshots exist, falling back to config."""
    m_discover = mocker.patch("src.data.discover_symbols", return_value=[])
    m_fetch = mocker.patch("src.data.fetch_and_snapshot", return_value=[])

    refresh_market_data(test_config, Console())

    m_discover.assert_called_once_with(test_config)
    # Comes from FULL_CONFIG_DICT's include_symbols list
    m_fetch.assert_called_once_with(
        ["TEST.NS", "INVALID.NS", "MISSING.NS"], test_config
    )


def test_refresh_market_data_no_symbols_anywhere(mocker: MockerFixture, test_config: Config) -> None:
    """Tests refresh when no snapshots or config symbols exist."""
    m_discover = mocker.patch("src.data.discover_symbols", return_value=[])
    m_fetch = mocker.patch("src.data.fetch_and_snapshot", return_value=[])

    # Create a config with no include_symbols
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["universe"]["include_symbols"] = []
    empty_config = cast(Config, _from_dict(Config, config_dict))

    refresh_market_data(empty_config, Console())

    m_discover.assert_called_once_with(empty_config)
    m_fetch.assert_not_called()
