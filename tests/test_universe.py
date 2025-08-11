"""
Comprehensive tests for universe selection system.

Tests cover:
1. NSE symbol loading and validation
2. Turnover calculation accuracy
3. Universe selection with various configurations
4. Filtering mechanisms (price, turnover, data quality)
5. Edge cases and error handling
6. Deterministic behavior validation
7. Survivorship bias handling
8. Integration tests with realistic market data patterns
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


class TestNSESymbols:
    """Test NSE symbol loading functionality."""
    
    def test_get_nse_symbols_returns_list(self):
        """Test that get_nse_symbols returns a list."""
        symbols = get_nse_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) > 0
    
    def test_get_nse_symbols_contains_expected_stocks(self):
        """Test that symbol list contains expected large-cap stocks."""
        symbols = get_nse_symbols()
        
        # Check for some well-known large-cap stocks
        expected_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        for stock in expected_stocks:
            assert stock in symbols
    
    def test_get_nse_symbols_no_duplicates(self):
        """Test that symbol list contains no duplicates."""
        symbols = get_nse_symbols()
        assert len(symbols) == len(set(symbols))
    
    def test_get_nse_symbols_sorted(self):
        """Test that symbols are returned in sorted order."""
        symbols = get_nse_symbols()
        assert symbols == sorted(symbols)


class TestTurnoverStats:
    """Test turnover calculation functionality."""
    
    def test_compute_turnover_stats_empty_data(self):
        """Test turnover stats with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['Close', 'Volume'])
        stats = compute_turnover_stats(empty_df)
        
        assert stats["median_turnover"] == 0.0
        assert stats["mean_turnover"] == 0.0
        assert stats["valid_days"] == 0
    
    def test_compute_turnover_stats_basic_calculation(self):
        """Test basic turnover calculation."""
        # Create test data: 5 days of trading
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Close': [100.0, 105.0, 102.0, 108.0, 110.0],
            'Volume': [1000, 1200, 800, 1500, 2000]
        }, index=dates)
        
        stats = compute_turnover_stats(data)
        
        # Expected turnovers: 100*1000=100k, 105*1200=126k, etc.
        expected_turnovers = [100000.0, 126000.0, 81600.0, 162000.0, 220000.0]
        
        assert stats["median_turnover"] == np.median(expected_turnovers)
        assert stats["mean_turnover"] == np.mean(expected_turnovers)
        assert stats["valid_days"] == 5
    
    def test_compute_turnover_stats_zero_volume_days(self):
        """Test handling of zero volume days."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Close': [100.0, 105.0, 102.0, 108.0, 110.0],
            'Volume': [1000, 0, 800, 0, 2000]  # Some zero volume days
        }, index=dates)
        
        stats = compute_turnover_stats(data)
        
        # Should exclude zero volume days
        assert stats["valid_days"] == 3
        assert stats["median_turnover"] > 0
    
    def test_compute_turnover_stats_lookback_filtering(self):
        """Test lookback period filtering."""
        
        # Create 3 years of data
        dates = pd.date_range('2020-01-01', periods=756, freq='D')  # ~3 years
        data = pd.DataFrame({
            'Close': [100.0] * 756,
            'Volume': [1000] * 756
        }, index=dates)
        
        # Test 1 year lookback
        stats_1yr = compute_turnover_stats(data, lookback_years=1)
        assert stats_1yr["valid_days"] <= 252  # ~252 trading days per year
        
        # Test 2 year lookback
        stats_2yr = compute_turnover_stats(data, lookback_years=2)
        assert stats_2yr["valid_days"] <= 504
    
    def test_compute_turnover_stats_all_zero_volume(self):
        """Test handling when all volume is zero."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Close': [100.0] * 5,
            'Volume': [0] * 5
        }, index=dates)
        
        stats = compute_turnover_stats(data)
        
        assert stats["median_turnover"] == 0.0
        assert stats["mean_turnover"] == 0.0
        assert stats["valid_days"] == 0


class TestUniverseSelector:
    """Test UniverseSelector class."""
    
    def test_universe_selector_initialization(self):
        """Test UniverseSelector initialization."""
        config = {
            "universe": {
                "size": 20,
                "min_turnover": 5000000,
                "min_price": 15.0,
                "exclude_symbols": ["TEST.NS"],
                "lookback_years": 1
            }
        }
        
        selector = UniverseSelector(config)
        
        assert selector.size == 20
        assert selector.min_turnover == 5000000
        assert selector.min_price == 15.0
        assert "TEST.NS" in selector.exclude_symbols
        assert selector.lookback_years == 1
    
    def test_universe_selector_validation(self):
        """Test configuration validation."""
        # Test invalid size
        with pytest.raises(ValueError, match="Universe size must be positive"):
            UniverseSelector({"universe": {"size": 0}})
            
        # Test invalid turnover
        with pytest.raises(ValueError, match="Minimum turnover must surpass base market activity requirements"):
            UniverseSelector({"universe": {"min_turnover": -1}})
            
        # Test invalid price
        with pytest.raises(ValueError, match="Minimum price must be non-negative"):
            UniverseSelector({"universe": {"min_price": -5}})
            
        # Test invalid lookback
        with pytest.raises(ValueError, match="Lookback years must be positive"):
            UniverseSelector({"universe": {"lookback_years": 0}})
    
    def test_apply_filters_price_filter(self):
        """Test price filtering in apply_filters."""
        config = {"universe": {"min_price": 50.0}}
        selector = UniverseSelector(config)
        
        # Create data with price below threshold
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'Close': [45.0] * 10,  # Below threshold
            'Volume': [1000] * 10
        }, index=dates)
        
        passes, reason = selector.apply_filters("TEST.NS", data)
        assert not passes
        assert "Price 45.00 < 50.00" in reason
    
    def test_apply_filters_turnover_filter(self):
        """Test turnover filtering in apply_filters."""
        config = {"universe": {"min_turnover": 1000000}}
        selector = UniverseSelector(config)
        
        # Create data with higher turnover to properly test filter (should fail)
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        close_prices = [100.0] * 500  # Increase price to create higher turnover for proper testing
        volumes = [100] * 500  # Very low volume
        
        data = pd.DataFrame({
            'Close': close_prices,
            'Volume': volumes
        }, index=dates)
        
        passes, reason = selector.apply_filters("TEST.NS", data)
        assert not passes
        assert "Turnover" in reason
    
    def test_apply_filters_insufficient_data(self):
        """Test filtering for insufficient data."""
        config = {"universe": {"lookback_years": 2}}
        selector = UniverseSelector(config)
        
        # Create very short data
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'Close': [100.0] * 5,
            'Volume': [1000] * 5
        }, index=dates)
        
        passes, reason = selector.apply_filters("TEST.NS", data)
        assert not passes
        assert "Insufficient data" in reason
    
    def test_apply_filters_passes_all(self):
        """Test symbol passing all filters."""
        config = {
            "universe": {
                "min_price": 10.0,
                "min_turnover": 1_000_000, # Updated to meet new validation
                "lookback_years": 1
            }
        }
        selector = UniverseSelector(config)
        
        # Create good data
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        data = pd.DataFrame({
            'Close': [100.0] * 252,
            'Volume': [10000] * 252  # High turnover
        }, index=dates)
        
        passes, reason = selector.apply_filters("GOOD.NS", data)
        assert passes
        assert reason == "Passes all filters"


class TestSelectUniverse:
    """Test main universe selection function."""
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_basic(self, mock_load, mock_get_symbols):
        """Test basic universe selection."""
        # Mock data for 3 symbols with sufficient days
        mock_data = {
            "RELIANCE.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "TCS.NS": pd.DataFrame({
                'Close': [200.0] * 400,
                'Volume': [50000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "INFY.NS": pd.DataFrame({
                'Close': [50.0] * 400,
                'Volume': [200000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {
            "universe": {
                "size": 2,
                "min_turnover": 1000000,
                "min_price": 10.0
            }
        }
        
        symbols, metadata = select_universe(config)
        
        assert len(symbols) == 2
        assert isinstance(metadata, dict)
        assert "selection_criteria" in metadata
        assert "symbol_details" in metadata
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_deterministic_ranking(self, mock_load, mock_get_symbols):
        """Test that ranking is deterministic."""
        
        # Create data with known turnover values
        mock_data = {
            "LOW.NS": pd.DataFrame({
                'Close': [10.0] * 400,
                'Volume': [100_000] * 400  # Turnover: 1,000,000 (1M)
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "MED.NS": pd.DataFrame({
                'Close': [50.0] * 400,
                'Volume': [30_000] * 400  # Turnover: 1,500,000 (1.5M)
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "HIGH.NS":pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [20_000] * 400  # Turnover: 2,000,000 (2M)
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {"universe": {"size": 3, "min_turnover": 1_000_000}}
        
        symbols1, _ = select_universe(config)
        symbols2, _ = select_universe(config)
        
        # Should be identical due to deterministic ranking
        assert symbols1 == symbols2
        assert symbols1 == ["HIGH.NS", "MED.NS", "LOW.NS"]  # Ranked by turnover
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_price_filtering(self, mock_load, mock_get_symbols):
        """Test penny stock filtering."""
        mock_data = {
            "PENNY.NS": pd.DataFrame({
                'Close': [5.0] * 400,  # Below threshold
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "GOOD.NS": pd.DataFrame({
                'Close': [100.0] * 400,  # Above threshold
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {
            "universe": {
                "size": 10,
                "min_price": 10.0  # Filter out penny stocks
            }
        }
        
        symbols, metadata = select_universe(config)
        
        assert "GOOD.NS" in symbols
        assert "PENNY.NS" not in symbols
        assert metadata["processing_stats"]["filtered_by_price"] >= 1
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_turnover_filtering(self, mock_load, mock_get_symbols):
        """Test illiquid stock filtering."""
        
        mock_data = {
            "ILLIQUID.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100] * 400  # Very low volume
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "LIQUID.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400  # High volume
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {
            "universe": {
                "size": 10,
                "min_turnover": 1000000  # ₹10L minimum
            }
        }
        
        symbols, metadata = select_universe(config)
        
        assert "LIQUID.NS" in symbols
        assert "ILLIQUID.NS" not in symbols
        assert metadata["processing_stats"]["filtered_by_turnover"] >= 1
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_exclusion_list(self, mock_load, mock_get_symbols):
        """Test exclusion list handling."""
        mock_data = {
            "EXCLUDE.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "INCLUDE.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {
            "universe": {
                "size": 10,
                "exclude_symbols": ["EXCLUDE.NS"]
            }
        }
        
        symbols, metadata = select_universe(config)
        
        assert "INCLUDE.NS" in symbols
        assert "EXCLUDE.NS" not in symbols
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_t0_filtering(self, mock_load, mock_get_symbols):
        """Test universe freezing at t0."""
        # Create data with different periods
        mock_data = {
            "OLD.NS": pd.DataFrame({
                'Close': [100.0] * 200,
                'Volume': [100000] * 200
            }, index=pd.date_range('2021-01-01', periods=200, freq='D')),
            "NEW.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {"universe": {"size": 10}}
        t0 = date(2022, 12, 31)
        
        symbols, metadata = select_universe(config, t0=t0)
        
        # OLD.NS should be excluded due to t0 filtering
        assert "OLD.NS" not in symbols
        assert metadata["selection_date"] == "2022-12-31"
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_insufficient_data(self, mock_load, mock_get_symbols):
        """Test handling when no stocks meet criteria."""
        mock_data = {
            "BAD1.NS": pd.DataFrame({
                'Close': [1.0] * 400,  # Penny stock
                'Volume': [100] * 400   # Illiquid
            }, index=pd.date_range('2022-01-01', periods=400, freq='D')),
            "BAD2.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100] * 400   # Illiquid
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {
            "universe": {
                "size": 10,
                "min_turnover": 1000000,
                "min_price": 50.0
            }
        }
        
        with pytest.raises(ValueError, match="No stocks met the universe selection criteria."):
            select_universe(config)
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_metadata_completeness(self, mock_load, mock_get_symbols):
        """Test that metadata contains all expected fields."""
        mock_data = {
            "TEST.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))}
    
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {"universe": {"size": 1}}
        
        symbols, metadata = select_universe(config)
        
        # Check all expected metadata fields
        # Check top-level fields
        top_level_fields = [
            "selection_criteria", "processing_stats", "symbol_details",
            "detailed_exclusions", "survivorship_bias_notes", "selection_date"
        ]
        for field in top_level_fields:
            assert field in metadata

        # Check fields within processing_stats
        processing_stats_fields = [
            "total_candidates", "missing_data", "empty_data",
            "filtered_by_price", "filtered_by_turnover", "filtered_by_days",
            "qualified_symbols", "selected_symbols"
        ]
        for field in processing_stats_fields:
            assert field in metadata["processing_stats"]

        # Check that detailed_exclusions is a list
        assert isinstance(metadata["detailed_exclusions"], list)
        
        # Check symbol details
        assert "TEST.NS" in metadata["symbol_details"]
        assert "median_turnover" in metadata["symbol_details"]["TEST.NS"]
        assert "rank" in metadata["symbol_details"]["TEST.NS"]
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_select_universe_survivorship_bias_documentation(self, mock_load, mock_get_symbols):
        """Test survivorship bias documentation in metadata."""
        mock_data = {
            "TEST.NS": pd.DataFrame({
                'Close': [100.0] * 400,
                'Volume': [100000] * 400
            }, index=pd.date_range('2022-01-01', periods=400, freq='D'))
        }
        mock_load.return_value = mock_data
        mock_get_symbols.return_value = list(mock_data.keys())
        
        config = {"universe": {"size": 1}}
        
        symbols, metadata = select_universe(config)
        
        # Check survivorship bias notes
        assert "survivorship_bias_notes" in metadata
        notes = metadata["survivorship_bias_notes"]
        
        # Should contain key survivorship bias mitigation strategies
        assert "frozen" in notes.lower()
        assert "historical" in notes.lower()
        assert "delisted" in notes.lower()


class TestIntegration:
    """Integration tests for complete universe selection workflow."""
    
    @patch('src.universe.get_nse_symbols')
    @patch('src.universe.load_snapshots')
    def test_full_workflow_with_realistic_data(self, mock_load, mock_get_symbols):
        """Test complete workflow with realistic market data patterns."""
        # Create realistic market data with different characteristics
        np.random.seed(42)  # Ensure reproducible test
        
        mock_data = {}
        
        # Large cap stocks - high price, high volume
        for symbol in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]:
            dates = pd.date_range('2021-01-01', periods=500, freq='D')
            close_prices = 2000 + np.random.normal(0, 100, 500)
            volumes = 1000000 + np.random.normal(0, 200000, 500)
            
            mock_data[symbol] = pd.DataFrame({
                'Close': np.maximum(close_prices, 1000),  # Ensure positive prices
                'Volume': np.maximum(volumes, 100000)    # Ensure positive volumes
            }, index=dates)
        
        # Small cap stocks - lower price, lower volume
        for symbol in ["SMALL1.NS", "SMALL2.NS"]:
            dates = pd.date_range('2021-01-01', periods=500, freq='D')
            close_prices = 50 + np.random.normal(0, 10, 500)
            volumes = 10000 + np.random.normal(0, 5000, 500)
            
            mock_data[symbol] = pd.DataFrame({
                'Close': np.maximum(close_prices, 10),
                'Volume': np.maximum(volumes, 1000)
            }, index=dates)
        
        # Penny stocks - very low price
        for symbol in ["PENNY1.NS", "PENNY2.NS"]:
            dates = pd.date_range('2021-01-01', periods=500, freq='D')
            close_prices = 2 + np.random.normal(0, 1, 500)
            volumes = 5000 + np.random.normal(0, 2000, 500)
            
            mock_data[symbol] = pd.DataFrame({
                'Close': np.maximum(close_prices, 1),
                'Volume': np.maximum(volumes, 1000)
            }, index=dates)
        
        mock_load.return_value = mock_data
        
        config = {
            "universe": {
                "size": 3,
                "min_turnover": 10000000,  # ₹1cr
                "min_price": 10.0,
                "lookback_years": 1,
                "exclude_symbols": ["SMALL2.NS"]
            }
        }
        
        mock_get_symbols.return_value = list(mock_data.keys())
        symbols, metadata = select_universe(config, t0=date(2023, 1, 1))
        
        # Should select large caps, exclude penny stocks and exclusions
        assert len(symbols) == 3
        assert all(symbol in symbols for symbol in ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
        assert "PENNY1.NS" not in symbols
        assert "SMALL2.NS" not in symbols  # Excluded
        assert metadata["processing_stats"]["filtered_by_price"] >= 2  # Penny stocks filtered
        assert "detailed_exclusions" in metadata
