"""
Additional comprehensive tests for universe selection edge cases.

This module focuses on:
1. Edge cases in turnover calculation
2. Error handling for malformed data
3. Boundary conditions in filtering
4. T0 filtering edge cases
5. Exclusion processing validation
6. Deterministic behavior under various conditions
"""

import pytest
from datetime import date, timedelta
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.universe import (
    select_universe, 
    compute_turnover_stats, 
    get_nse_symbols,
    UniverseSelector
)


class TestTurnoverStatsEdgeCases:
    """Test edge cases in turnover calculation."""
    
    def test_compute_turnover_stats_negative_values(self):
        """Test handling of negative values in data."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Close': [100.0, -90.0, 80.0, -75.0, 70.0],
            'Volume': [1000, 2000, 1500, 3000, 2500]
        }, index=dates)
        
        turnover = compute_turnover_stats(data)
        assert isinstance(turnover, dict), "Turnover should return a dictionary"
        assert turnover['median_turnover'] >= 0, "Median turnover should be non-negative"
        assert turnover['mean_turnover'] >= 0, "Mean turnover should be non-negative"
    
    def test_compute_turnover_stats_extreme_values(self):
        """Test handling of extreme values."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Close': [1e-6, 1e6, 1.0, 1e-3, 1e9],  # Very small to very large
            'Volume': [1e-6, 1e6, 1.0, 1e3, 1e9]
        }, index=dates)
        
        stats = compute_turnover_stats(data)
        
        # Should handle extreme values without overflow
        assert np.isfinite(stats["median_turnover"])
        assert np.isfinite(stats["mean_turnover"])
    
    def test_compute_turnover_stats_missing_columns(self):
        """Test handling of missing required columns."""
        # Create data with missing 'turnover' column
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'price': range(1,6)
        })
        
        # Should raise ValueError for missing columns
        with pytest.raises(ValueError, match="Missing required columns: Close, Volume"):
            compute_turnover_stats(data)
        
        # Missing Close column
        data_missing_close = pd.DataFrame({
            'Volume': [1000] * 5
        }, index=dates)
        
        with pytest.raises(ValueError, match="Missing required columns: Close"):
            compute_turnover_stats(data_missing_close)
        data_missing_volume = pd.DataFrame({
            'Close': [100.0] * 5
        }, index=dates)
        
        with pytest.raises(ValueError, match="Missing required columns: Volume"):
            compute_turnover_stats(data_missing_volume)
    
    def test_compute_turnover_stats_single_day(self):
        """Test with single day of data."""
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        data = pd.DataFrame({
            'Close': [100.0],
            'Volume': [1000]
        }, index=dates)
        
        stats = compute_turnover_stats(data)
        
        assert stats["median_turnover"] == 100000.0
        assert stats["mean_turnover"] == 100000.0
        assert stats["valid_days"] == 1
    
    def test_compute_turnover_stats_zero_lookback(self):
        """Test with zero lookback years."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'Close': [100.0] * 100,
            'Volume': [1000] * 100
        }, index=dates)
        
        stats = compute_turnover_stats(data, lookback_years=0)
        
        # Should use all available data
        assert stats["valid_days"] == 100


class TestUniverseSelectorEdgeCases:
    """Test edge cases in UniverseSelector."""
    
    def test_universe_selector_empty_config(self):
        """Test with empty configuration."""
        selector = UniverseSelector({})
        
        # Should use defaults
        assert selector.size == 10
        assert selector.min_turnover == 10000000.0
        assert selector.min_price == 10.0
        assert selector.exclude_symbols == []
        assert selector.lookback_years == 2
    
    def test_universe_selector_partial_config(self):
        """Test with partial configuration."""
        config = {
            "universe": {
                "size": 25,
                "min_price": 20.0
                # Missing other parameters
            }
        }
        
        selector = UniverseSelector(config)
        
        assert selector.size == 25
        assert selector.min_price == 20.0
        assert selector.min_turnover == 10000000.0  # Default
        assert selector.lookback_years == 2  # Default
    
    def test_universe_selector_non_list_exclusions(self):
        """Test handling of non-list exclusions."""
        config = {
            "universe": {
                "exclude_symbols": "INVALID"  # Should be list
            }
        }
        
        with pytest.raises(ValueError, match="exclude_symbols must be a list"):
            selector = UniverseSelector(config)
            selector.validate_config()
    
    def test_apply_filters_empty_dataframe(self):
        """Test apply_filters with empty DataFrame."""
        selector = UniverseSelector({"universe": {}})
        
        empty_df = pd.DataFrame(columns=['Close', 'Volume'])
        passes, reason = selector.apply_filters("TEST.NS", empty_df)
        
        assert not passes
        assert reason == "Empty data"
    
    def test_apply_filters_single_row(self):
        """Test apply_filters with single row of data."""
        selector = UniverseSelector({
            "universe": {
                "min_price": 50.0,
                "min_turnover": 1000000,
                "lookback_years": 1
            }
        })
        
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        data = pd.DataFrame({
            'Close': [100.0],
            'Volume': [2000]
        }, index=dates)
        
        passes, reason = selector.apply_filters("TEST.NS", data)
        
        # Single day might not meet minimum days requirement
        assert not passes
        assert "Insufficient data" in reason


class TestSelectUniverseEdgeCases:
    """Test edge cases in select_universe function."""

    @patch('src.universe.load_snapshots')
    def test_select_universe_no_data_available(self, mock_load):
        """Test when no data is available for any symbol."""
        mock_load.return_value = {}
        config = {"universe": {"size": 10}}
        with pytest.raises(ValueError, match="Cannot select universe"):
            select_universe(config, available_symbols=["DUMMY.NS"])

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_all_filtered_out(self, mock_load, mock_get_symbols):
        """Test when all symbols are filtered out."""
        mock_data = {
            "PENNY1.NS": pd.DataFrame({
                'Close': [1.0] * 400,
                'Volume': [100] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
        }
        mock_load.return_value = mock_data
        config = {
            "universe": {
                "size": 10,
                "min_price": 50.0,
                "min_turnover": 1_000_000,
                "lookback_years": 1
            }
        }
        with pytest.raises(ValueError, match="No stocks met the universe selection criteria."):
            select_universe(config, available_symbols=list(mock_data.keys()))

    @patch('src.universe.load_snapshots')
    def test_select_universe_t0_before_all_data(self, mock_load):
        """Test t0 filtering when t0 is before all available data."""
        mock_data = {
            "TEST.NS": pd.DataFrame({
                'Close': [100.0] * 200,
                'Volume': [100000] * 200
            }, index=pd.date_range('2023-01-01', periods=200, freq='D'))
        }
        mock_load.return_value = mock_data
        config = {"universe": {"size": 10, "lookback_years": 1}}
        t0 = date(2022, 1, 1)
        with pytest.raises(ValueError, match="No stocks met the universe selection criteria."):
            select_universe(config, t0=t0, available_symbols=list(mock_data.keys()))

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_t0_exact_match(self, mock_load, mock_get_symbols):
        """Test t0 filtering with exact date match."""
        mock_data = {
            "TEST.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        config = {"universe": {"size": 10, "lookback_years": 1}}
        t0 = date(2023, 1, 1)
        symbols, metadata = select_universe(config, t0=t0, available_symbols=list(mock_data.keys()))
        assert "TEST.NS" in symbols
        assert metadata["selection_date"] == str(t0)

    @patch('src.universe.load_snapshots')
    def test_select_universe_zero_size(self, mock_load):
        """Test with zero universe size."""
        with pytest.raises(ValueError, match="Universe size must be positive"):
            select_universe({"universe": {"size": 0}})

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_larger_than_available(self, mock_load, mock_get_symbols):
        """Test when requested size is larger than available stocks."""
        mock_data = {
            "TEST1.NS": pd.DataFrame({
                'Close': [100.0] * 200, 'Volume': [100000] * 200
            }, index=pd.date_range('2023-01-01', periods=200, freq='D')),
            "TEST2.NS": pd.DataFrame({
                'Close': [200.0] * 200, 'Volume': [50000] * 200
            }, index=pd.date_range('2023-01-01', periods=200, freq='D'))
        }
        mock_load.return_value = mock_data
        config = {"universe": {"size": 10, "lookback_years": 0.5}}
        symbols, metadata = select_universe(config, available_symbols=list(mock_data.keys()))
        assert len(symbols) == 2
        assert "TEST1.NS" in symbols and "TEST2.NS" in symbols

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_all_excluded(self, mock_load, mock_get_symbols):
        """Test when all symbols are in exclusion list."""
        mock_data = {"DUMMY.NS": pd.DataFrame()}
        mock_load.return_value = mock_data
        config = {"universe": {"exclude_symbols": ["DUMMY.NS"]}}
        with pytest.raises(ValueError, match="No stocks met the universe selection criteria."):
            select_universe(config, available_symbols=["DUMMY.NS"])

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_deterministic_with_t0(self, mock_load, mock_get_symbols):
        """Test deterministic behavior with t0 filtering."""
        mock_data = {
            "EARLY.NS": pd.DataFrame({
                'Close': [100.0] * 400, 'Volume': [100000] * 400
            }, index=pd.date_range('2021-01-01', periods=400, freq='D')),
            "LATE.NS": pd.DataFrame({
                'Close': [200.0] * 200, 'Volume': [50000] * 200
            }, index=pd.date_range('2023-01-01', periods=200, freq='D'))
        }
        mock_load.return_value = mock_data
        config = {"universe": {"size": 10, "lookback_years": 1}}
        t0 = date(2022, 12, 31)
        
        symbols1, metadata1 = select_universe(config, t0=t0, available_symbols=list(mock_data.keys()))
        symbols2, metadata2 = select_universe(config, t0=t0, available_symbols=list(mock_data.keys()))
        
        assert symbols1 == ["EARLY.NS"]
        assert symbols1 == symbols2
        assert metadata1 == metadata2
        with pytest.raises(ValueError, match="No stocks met the universe selection criteria."):
            select_universe(config, t0=date(2020, 1, 1), available_symbols=list(mock_data.keys()))

    @patch('src.universe.load_snapshots')
    def test_select_universe_metadata_detailed_exclusions(self, mock_load):
        """Test detailed exclusion tracking in metadata."""
        mock_data = {
            "PENNY.NS": pd.DataFrame({
                'Close': [5.0] * 200, 'Volume': [100000] * 200
            }, index=pd.date_range('2022-01-01', periods=200, freq='D')),
            "ILLIQUID.NS": pd.DataFrame({
                'Close': [100.0] * 200, 'Volume': [100] * 200
            }, index=pd.date_range('2022-01-01', periods=200, freq='D')),
            "GOOD.NS": pd.DataFrame({
                'Close': [100.0] * 400, 'Volume': [1000000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        config = {
            "universe": {
                "size": 10, "min_price": 10.0, "min_turnover": 1_000_000, "lookback_years": 1
            }
        }
        symbols, metadata = select_universe(config, available_symbols=list(mock_data.keys()))
        assert symbols == ["GOOD.NS"]
        exclusions = metadata["detailed_exclusions"]
        assert any(e["symbol"] == "PENNY.NS" and "Price" in e["reason"] for e in exclusions)
        assert any(e["symbol"] == "ILLIQUID.NS" and "Turnover" in e["reason"] for e in exclusions)


class TestDeterministicBehavior:
    """Test deterministic behavior under various conditions."""

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_deterministic_with_different_data_orders(self, mock_load, mock_get_symbols):
        """Test that data order doesn't affect deterministic ranking."""
        data1 = pd.DataFrame({
            'Close': ([100.0, 200.0, 150.0] * 100),
            'Volume': ([1000, 2000, 1500] * 100)
        }, index=pd.date_range('2023-01-01', periods=300, freq='D'))
        data2 = pd.DataFrame({
            'Close': ([150.0, 100.0, 200.0] * 100),
            'Volume': ([1500, 1000, 2000] * 100)
        }, index=pd.date_range('2023-01-03', periods=300, freq='D'))
        mock_data = {"SYMBOL1.NS": data1, "SYMBOL2.NS": data2}
        mock_load.return_value = mock_data
        config = {"universe": {
            "size": 10, "lookback_years": 1, "min_turnover": 0, "min_price": 0
        }}
        symbols1, metadata1 = select_universe(config, available_symbols=list(mock_data.keys()))
        mock_load.return_value = mock_data
        symbols2, metadata2 = select_universe(config, available_symbols=list(mock_data.keys()))
        assert symbols1 == symbols2
        assert metadata1 == metadata2
        assert len(symbols1) == 2

    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_deterministic_with_floating_point_precision(self, mock_load, mock_get_symbols):
        """Test deterministic behavior with floating point calculations."""
        mock_data = {}
        for i, symbol in enumerate(["A.NS", "B.NS", "C.NS"]):
            base_turnover = 100000.0 + i * 1e-4
            dates = pd.date_range('2023-01-01', periods=200, freq='D')
            data = pd.DataFrame({
                'Close': [base_turnover / 1000] * 200,
                'Volume': [1000] * 200
            }, index=dates)
            mock_data[symbol] = data
        mock_load.return_value = mock_data
        config = {"universe": {
            "size": 10, "lookback_years": 0.5, "min_turnover": 0, "min_price": 0
        }}
        results = []
        for _ in range(5):
            symbols, metadata = select_universe(config, available_symbols=list(mock_data.keys()))
            results.append((symbols, metadata))
            assert len(symbols) == 3
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result
