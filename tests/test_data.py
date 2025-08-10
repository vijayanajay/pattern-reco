"""
Tests for data fetching and snapshot management.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data import fetch_and_snapshot, load_snapshots


class TestDataFetching:
    """Test data fetching functionality."""
    
    @patch('yfinance.Ticker')
    def test_fetch_and_snapshot_success(self, mock_ticker_class):
        """Successful data fetching should save snapshots."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock yfinance Ticker
            mock_ticker = Mock()
            mock_data = pd.DataFrame({
                'Open': [100.0, 101.0],
                'High': [102.0, 103.0],
                'Low': [99.0, 100.0],
                'Close': [101.0, 102.0],
                'Volume': [1000, 1100]
            }, index=pd.date_range('2023-01-01', periods=2))
            
            mock_ticker.history.return_value = mock_data
            mock_ticker_class.return_value = mock_ticker
            
            config = {
                "data": {
                    "interval": "1d",
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-03"
                }
            }
            
            # Change to temp directory for test
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Should not raise exception
                fetch_and_snapshot(["TEST.NS"], config)
                
                # Verify ticker was called
                mock_ticker.history.assert_called_once()
                
                # Verify files were created
                snapshot_dir = Path(temp_dir) / "data" / "snapshots"
                assert (snapshot_dir / "TEST.NS.parquet").exists()
                
            finally:
                os.chdir(original_cwd)
    
    @patch('yfinance.Ticker')
    def test_fetch_and_snapshot_empty_data(self, mock_ticker_class):
        """Empty data should be handled gracefully."""
        # Mock ticker returning empty DataFrame
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        config = {
            "data": {
                "interval": "1d",
                "start_date": "2023-01-01",
                "end_date": "2023-01-03"
            }
        }
        
        # Should raise RuntimeError with failed symbols
        with pytest.raises(RuntimeError, match="Failed to fetch 1 symbols"):
            fetch_and_snapshot(["INVALID.NS"], config)
        
        # Verify ticker was called
        mock_ticker.history.assert_called_once()


class TestSnapshotManagement:
    """Test snapshot loading functionality."""
    
    def test_load_snapshots_missing_directory(self):
        """Loading from non-existent directory should raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                # No data directory exists in temp dir
                with pytest.raises(FileNotFoundError, match="Snapshot directory not found"):
                    load_snapshots(["TEST.NS"])
            finally:
                os.chdir(original_cwd)
    
    def test_load_snapshots_missing_file(self):
        """Loading missing snapshot should raise error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create snapshot directory but no files
            snapshot_dir = Path(temp_dir) / "data" / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                with pytest.raises(FileNotFoundError, match="Snapshot not found"):
                    load_snapshots(["TEST.NS"])
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__])