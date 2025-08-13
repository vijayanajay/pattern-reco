"""
Tests for data fetching and snapshot management.
"""
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from rich.console import Console

from src.config import Config
from src.data import fetch_and_snapshot, load_snapshots


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Pytest fixture to create a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    """Pytest fixture to create a valid Config object for testing."""
    config_dict = {
        "run": {"name": "test_run"},
        "data": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "source": "yfinance_test",
            "interval": "1d",
            "snapshot_dir": str(temp_dir),
        },
        "universe": {"include_symbols": ["TEST.NS", "INVALID.NS", "MISSING.NS"]},
        "detector": {}, "walk_forward": {}, "execution": {}, "portfolio": {}, "reporting": {},
    }
    # A simplified config for these tests
    return Config.parse_obj(config_dict)


@pytest.fixture
def console() -> Console:
    """Fixture for a Rich console that discards output."""
    return Console(file=open(Path.cwd() / "test.log", "w"))


@patch("yfinance.Ticker")
def test_fetch_and_snapshot_success(
    mock_ticker_class: Mock, test_config: Config, console: Console
) -> None:
    """Test that successful data fetching saves a snapshot."""
    mock_ticker = Mock()
    mock_data = pd.DataFrame({"Close": [100.0]})
    mock_ticker.history.return_value = mock_data
    mock_ticker_class.return_value = mock_ticker

    fetch_and_snapshot(["TEST.NS"], test_config, console)

    expected_path = (
        Path(test_config.data.snapshot_dir)
        / f"{test_config.data.source}_{test_config.data.interval}"
        / "TEST.NS.parquet"
    )
    assert expected_path.exists()
    df = pd.read_parquet(expected_path)
    assert df["Close"].iloc[0] == 100.0


@patch("yfinance.Ticker")
def test_fetch_and_snapshot_empty_data(
    mock_ticker_class: Mock, test_config: Config, console: Console
) -> None:
    """Test that empty data from yfinance does not create a file."""
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame()  # Empty data
    mock_ticker_class.return_value = mock_ticker

    fetch_and_snapshot(["INVALID.NS"], test_config, console)

    expected_path = (
        Path(test_config.data.snapshot_dir)
        / f"{test_config.data.source}_{test_config.data.interval}"
        / "INVALID.NS.parquet"
    )
    assert not expected_path.exists()


def test_load_snapshots_success(test_config: Config, console: Console) -> None:
    """Test successfully loading an existing snapshot."""
    snapshot_subdir = (
        Path(test_config.data.snapshot_dir)
        / f"{test_config.data.source}_{test_config.data.interval}"
    )
    snapshot_subdir.mkdir()
    fake_snapshot_path = snapshot_subdir / "TEST.NS.parquet"
    pd.DataFrame({"Close": [100.0]}).to_parquet(fake_snapshot_path)

    data = load_snapshots(["TEST.NS"], test_config, console)
    assert "TEST.NS" in data
    assert isinstance(data["TEST.NS"], pd.DataFrame)
    assert data["TEST.NS"]["Close"].iloc[0] == 100.0


def test_load_snapshots_missing_directory_raises_error(
    test_config: Config, console: Console
) -> None:
    """Test that loading from a non-existent snapshot directory raises an error."""
    test_config.data.snapshot_dir = str(Path(test_config.data.snapshot_dir) / "non_existent")
    with pytest.raises(FileNotFoundError, match="Snapshot directory not found"):
        load_snapshots(["TEST.NS"], test_config, console)


def test_load_snapshots_missing_file_is_ignored(
    test_config: Config, console: Console
) -> None:
    """Test that loading a missing snapshot file is ignored and does not raise an error."""
    snapshot_subdir = (
        Path(test_config.data.snapshot_dir)
        / f"{test_config.data.source}_{test_config.data.interval}"
    )
    snapshot_subdir.mkdir()

    # The function should not raise an error, just return an empty dict
    data = load_snapshots(["MISSING.NS"], test_config, console)
    assert not data # Expect an empty dictionary
