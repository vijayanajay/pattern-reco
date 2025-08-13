"""
Tests for data fetching and snapshot management from the yfinance adapter.
"""
import logging
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture

from src.adapters.yfinance_api import fetch_and_snapshot, load_snapshots


@pytest.fixture
def temp_snapshot_dir() -> Generator[Path, None, None]:
    """Pytest fixture to create a temporary snapshot directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@patch('yfinance.Ticker')
def test_fetch_and_snapshot_success(mock_ticker_class: Mock, temp_snapshot_dir: Path) -> None:
    """Test that successful data fetching saves a snapshot."""
    mock_ticker = Mock()
    mock_data = pd.DataFrame({'Close': [100.0]})
    mock_ticker.history.return_value = mock_data
    mock_ticker_class.return_value = mock_ticker

    config = {
        "data": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-03",
            "interval": "1d",
            "source": "yfinance_test",
            "snapshot_dir": str(temp_snapshot_dir)
        }
    }

    fetch_and_snapshot(["TEST.NS"], config)

    expected_path = temp_snapshot_dir / "yfinance_test_1d" / "TEST.NS.parquet"
    assert expected_path.exists()
    df = pd.read_parquet(expected_path)
    assert df['Close'].iloc[0] == 100.0

@patch('yfinance.Ticker')
def test_fetch_and_snapshot_empty_data_logs_warning(mock_ticker_class: Mock, temp_snapshot_dir: Path, caplog: LogCaptureFixture) -> None:
    """Test that empty data from yfinance logs a warning and does not create a file."""
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame() # Empty data
    mock_ticker_class.return_value = mock_ticker

    config = {
        "data": {
            "start_date": "2023-01-01",
            "end_date": "2023-01-03",
            "interval": "1d",
            "source": "yfinance_test",
            "snapshot_dir": str(temp_snapshot_dir)
        }
    }

    with caplog.at_level(logging.WARNING):
        fetch_and_snapshot(["INVALID.NS"], config)

    assert "No data returned for INVALID.NS" in caplog.text
    expected_path = temp_snapshot_dir / "yfinance_test_1d" / "INVALID.NS.parquet"
    assert not expected_path.exists()

def test_load_snapshots_success(temp_snapshot_dir: Path) -> None:
    """Test successfully loading an existing snapshot."""
    # Create a fake snapshot
    snapshot_subdir = temp_snapshot_dir / "yfinance_test_1d"
    snapshot_subdir.mkdir()
    fake_snapshot_path = snapshot_subdir / "TEST.NS.parquet"
    pd.DataFrame({'Close': [100.0]}).to_parquet(fake_snapshot_path)

    config = {
        "data": {
            "interval": "1d",
            "source": "yfinance_test",
            "snapshot_dir": str(temp_snapshot_dir)
        }
    }

    data = load_snapshots(["TEST.NS"], config)
    assert "TEST.NS" in data
    assert isinstance(data["TEST.NS"], pd.DataFrame)
    assert data["TEST.NS"]['Close'].iloc[0] == 100.0

def test_load_snapshots_missing_directory_raises_error(temp_snapshot_dir: Path) -> None:
    """Test that loading from a non-existent snapshot directory raises an error."""
    config = {
        "data": {
            "interval": "1d",
            "source": "yfinance_test",
            "snapshot_dir": str(temp_snapshot_dir / "non_existent_dir")
        }
    }
    with pytest.raises(FileNotFoundError, match="Snapshot directory not found"):
        load_snapshots(["TEST.NS"], config)

def test_load_snapshots_missing_file_raises_error(temp_snapshot_dir: Path) -> None:
    """Test that loading a missing snapshot file raises an error."""
    snapshot_subdir = temp_snapshot_dir / "yfinance_test_1d"
    snapshot_subdir.mkdir()

    config = {
        "data": {
            "interval": "1d",
            "source": "yfinance_test",
            "snapshot_dir": str(temp_snapshot_dir)
        }
    }
    with pytest.raises(FileNotFoundError, match="Snapshot not found for MISSING.NS"):
        load_snapshots(["MISSING.NS"], config)
