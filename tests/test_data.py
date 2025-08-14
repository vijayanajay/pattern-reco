"""
Tests for data fetching and snapshot management.
"""
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.config import Config
from src.data import fetch_and_snapshot, load_snapshots


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Pytest fixture to create a valid Config object for testing."""
    config_dict = {
        "run": {"name": "test_run", "t0": "2023-01-15"},
        "data": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "source": "yfinance_test",
            "interval": "1d",
            "snapshot_dir": str(tmp_path),
        },
        "universe": {"include_symbols": ["TEST.NS", "INVALID.NS", "MISSING.NS"]},
        "detector": {}, "walk_forward": {}, "execution": {}, "portfolio": {}, "reporting": {},
    }
    return Config.parse_obj(config_dict)


@patch("src.data.yf.download")
def test_fetch_and_snapshot_success(mock_download: Mock, test_config: Config) -> None:
    """Test that successful data fetching saves a snapshot."""
    mock_data = pd.DataFrame({"Close": [100.0], "Volume": [1000.0], "Open": [99.0], "High": [101.0], "Low": [98.0]})
    mock_download.return_value = mock_data

    failed = fetch_and_snapshot(["TEST.NS"], test_config)
    assert not failed

    snapshot_dir = test_config.data.snapshot_dir
    interval = test_config.data.interval
    source = test_config.data.source
    expected_path = snapshot_dir / f"{source}_{interval}" / "TEST.NS.parquet"

    assert expected_path.exists()
    df = pd.read_parquet(expected_path)
    assert df["Close"].iloc[0] == 100.0


@patch("src.data.yf.download")
def test_fetch_and_snapshot_empty_data(mock_download: Mock, test_config: Config) -> None:
    """Test that empty data from yfinance is treated as a failure."""
    mock_download.return_value = pd.DataFrame()

    failed = fetch_and_snapshot(["INVALID.NS"], test_config)
    assert failed == ["INVALID.NS"]

    snapshot_dir = test_config.data.snapshot_dir
    interval = test_config.data.interval
    source = test_config.data.source
    expected_path = snapshot_dir / f"{source}_{interval}" / "INVALID.NS.parquet"
    assert not expected_path.exists()


def test_load_snapshots_success_and_turnover_calc(test_config: Config) -> None:
    """Test loading a snapshot and that Turnover is correctly calculated."""
    snapshot_dir = test_config.data.snapshot_dir
    interval = test_config.data.interval
    source = test_config.data.source
    snapshot_subdir = snapshot_dir / f"{source}_{interval}"
    snapshot_subdir.mkdir()
    fake_snapshot_path = snapshot_subdir / "TEST.NS.parquet"
    pd.DataFrame({"Close": [100.0], "Volume": [1000.0], "Open": [99.0], "High": [101.0], "Low": [98.0]}).to_parquet(fake_snapshot_path)

    data = load_snapshots(["TEST.NS"], test_config)
    assert "TEST.NS" in data
    df = data["TEST.NS"]
    assert isinstance(df, pd.DataFrame)
    assert "Turnover" in df.columns
    assert df["Turnover"].iloc[0] == 100.0 * 1000.0


def test_load_snapshots_missing_directory_raises_error(test_config: Config) -> None:
    """Test that loading from a non-existent snapshot directory raises an error."""
    # To avoid modifying the fixture, we create a new config object with a modified path
    non_existent_path = test_config.data.snapshot_dir / "non_existent"
    test_config.data.snapshot_dir = non_existent_path

    with pytest.raises(FileNotFoundError, match="Snapshot directory not found"):
        load_snapshots(["TEST.NS"], test_config)


def test_load_snapshots_missing_file_raises_error(test_config: Config) -> None:
    """Test that loading a missing snapshot file now raises an error."""
    snapshot_dir = test_config.data.snapshot_dir
    interval = test_config.data.interval
    source = test_config.data.source
    snapshot_subdir = snapshot_dir / f"{source}_{interval}"
    snapshot_subdir.mkdir(exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Missing snapshot for symbol: MISSING.NS"):
        load_snapshots(["MISSING.NS"], test_config)
