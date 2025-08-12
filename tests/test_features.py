"""
Tests for feature engineering components.
"""

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.features import FeatureEngine

@pytest.fixture
def sample_ohlc_data():
    """Fixture for sample OHLC data."""
    return pd.DataFrame({
        'Open': [100, 102, 101, 103, 105],
        'High': [103, 104, 103, 106, 107],
        'Low': [99, 101, 100, 102, 104],
        'Close': [102, 101, 103, 105, 106],
        'Volume': [1000, 1200, 800, 1500, 2000]
    })

class TestFeatureEngine:

    def test_add_daily_returns(self, sample_ohlc_data):
        """Test daily returns calculation."""
        df = FeatureEngine.add_daily_returns(sample_ohlc_data)
        assert 'daily_return' in df.columns
        # First value should be NaN
        assert pd.isna(df['daily_return'].iloc[0])
        # (101 / 102) - 1 = -0.0098...
        assert np.isclose(df['daily_return'].iloc[1], (101/102)-1)

    def test_add_daily_returns_empty_df(self):
        """Test daily returns with empty DataFrame."""
        df = pd.DataFrame({'Close': []})
        df = FeatureEngine.add_daily_returns(df)
        assert 'daily_return' in df.columns
        assert df.empty

    def test_add_daily_returns_missing_column(self):
        """Test daily returns with missing 'Close' column."""
        with pytest.raises(ValueError, match="DataFrame must have 'Close' column"):
            FeatureEngine.add_daily_returns(pd.DataFrame({'Open': [1,2]}))

    def test_add_gap_percentage(self, sample_ohlc_data):
        """Test gap percentage calculation."""
        df = FeatureEngine.add_gap_percentage(sample_ohlc_data)
        assert 'gap_pct' in df.columns
        # First value should be NaN
        assert pd.isna(df['gap_pct'].iloc[0])
        # (102 - 102) / 102 = 0
        assert np.isclose(df['gap_pct'].iloc[1], (102-102)/102)
        # (101 - 101) / 101 = 0
        assert np.isclose(df['gap_pct'].iloc[2], (101-101)/101)

    def test_add_gap_percentage_empty_df(self):
        """Test gap percentage with empty DataFrame."""
        df = pd.DataFrame({'Open': [], 'Close': []})
        df = FeatureEngine.add_gap_percentage(df)
        assert 'gap_pct' in df.columns
        assert df.empty

    def test_add_gap_percentage_missing_column(self):
        """Test gap percentage with missing columns."""
        with pytest.raises(ValueError, match="DataFrame must have 'Open' and 'Close' columns"):
            FeatureEngine.add_gap_percentage(pd.DataFrame({'Close': [1,2]}))
        with pytest.raises(ValueError, match="DataFrame must have 'Open' and 'Close' columns"):
            FeatureEngine.add_gap_percentage(pd.DataFrame({'Open': [1,2]}))

    def test_add_rolling_zscore(self, sample_ohlc_data):
        """Test rolling z-score calculation."""
        df = sample_ohlc_data.copy()
        df['test_col'] = [1, 2, 1, 2, 3]
        window = 3
        df = FeatureEngine.add_rolling_zscore(df, 'test_col', window)

        col_name = f'test_col_zscore_{window}'
        assert col_name in df.columns

        # First two values should be NaN due to window size
        assert pd.isna(df[col_name].iloc[0])
        assert pd.isna(df[col_name].iloc[1])

        # Manual calculation for the first z-score at index 2
        # Window is [1, 2, 1]. Mean = 1.333, Std = 0.577
        # z-score = (1 - 1.333) / 0.577 = -0.577
        assert np.isclose(df[col_name].iloc[2], (1 - np.mean([1,2,1])) / np.std([1,2,1], ddof=1))

    def test_add_rolling_zscore_zero_std(self):
        """Test rolling z-score with zero standard deviation."""
        df = pd.DataFrame({'test_col': [1, 1, 1, 2, 3]})
        window = 3
        df = FeatureEngine.add_rolling_zscore(df, 'test_col', window)
        col_name = f'test_col_zscore_{window}'

        # z-score for window [1, 1, 1] should be NaN because std is 0
        assert pd.isna(df[col_name].iloc[2])
        # z-score for window [1, 1, 2] should be valid
        assert not pd.isna(df[col_name].iloc[3])

    def test_add_rolling_zscore_missing_column(self):
        """Test rolling z-score with missing column."""
        with pytest.raises(ValueError, match="Column 'non_existent' not in DataFrame"):
            FeatureEngine.add_rolling_zscore(pd.DataFrame({'A': [1]}), 'non_existent', 3)
