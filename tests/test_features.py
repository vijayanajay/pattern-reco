"""
Tests for feature engineering functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.core import features


@pytest.fixture
def sample_ohlc_data() -> pd.DataFrame:
    """Fixture for sample OHLC data."""
    return pd.DataFrame({
        'Open': [100, 102, 101, 103, 105],
        'High': [103, 104, 103, 106, 107],
        'Low': [99, 101, 100, 102, 104],
        'Close': [102, 101, 103, 105, 106],
        'Volume': [1000, 1200, 800, 1500, 2000]
    })

def test_add_daily_returns(sample_ohlc_data: pd.DataFrame) -> None:
    """Test daily returns calculation."""
    df = features.add_daily_returns(sample_ohlc_data)
    assert 'daily_return' in df.columns
    assert pd.isna(df['daily_return'].iloc[0])
    assert np.isclose(df['daily_return'].iloc[1], (101/102)-1)

def test_add_daily_returns_empty_df() -> None:
    """Test daily returns with empty DataFrame."""
    df = pd.DataFrame({'Close': []})
    df = features.add_daily_returns(df)
    assert 'daily_return' in df.columns
    assert df.empty

def test_add_daily_returns_missing_column() -> None:
    """Test daily returns with missing 'Close' column."""
    with pytest.raises(ValueError, match="must have a 'Close' column"):
        features.add_daily_returns(pd.DataFrame({'Open': [1,2]}))

def test_add_gap_percentage(sample_ohlc_data: pd.DataFrame) -> None:
    """Test gap percentage calculation."""
    df = features.add_gap_percentage(sample_ohlc_data)
    assert 'gap_pct' in df.columns
    assert pd.isna(df['gap_pct'].iloc[0])
    assert np.isclose(df['gap_pct'].iloc[1], (102-102)/102)
    assert np.isclose(df['gap_pct'].iloc[2], (101-101)/101)

def test_add_gap_percentage_empty_df() -> None:
    """Test gap percentage with empty DataFrame."""
    df = pd.DataFrame({'Open': [], 'Close': []})
    df = features.add_gap_percentage(df)
    assert 'gap_pct' in df.columns
    assert df.empty

def test_add_gap_percentage_missing_column() -> None:
    """Test gap percentage with missing columns."""
    with pytest.raises(ValueError, match="must have 'Open' and 'Close' columns"):
        features.add_gap_percentage(pd.DataFrame({'Close': [1,2]}))

def test_add_rolling_zscore(sample_ohlc_data: pd.DataFrame) -> None:
    """Test rolling z-score calculation."""
    df = sample_ohlc_data.copy()
    df['test_col'] = [1, 2, 1, 2, 3]
    window = 3
    df = features.add_rolling_zscore(df, 'test_col', window)

    col_name = f'test_col_zscore_{window}'
    assert col_name in df.columns
    assert pd.isna(df[col_name].iloc[0])
    # The z-score at index 1 is not NaN because min_periods=2 for window=3
    assert np.isclose(df[col_name].iloc[2], (1 - np.mean([1,2,1])) / np.std([1,2,1], ddof=1))

def test_add_rolling_zscore_zero_std() -> None:
    """Test rolling z-score with zero standard deviation."""
    df = pd.DataFrame({'test_col': [1, 1, 1, 2, 3]})
    window = 3
    df = features.add_rolling_zscore(df, 'test_col', window)
    col_name = f'test_col_zscore_{window}'
    assert pd.isna(df[col_name].iloc[2]) # z-score for [1, 1, 1] should be NaN
    assert not pd.isna(df[col_name].iloc[3])

def test_add_gap_z_signal() -> None:
    """Tests Gap-Z signal and score generation."""
    data = pd.DataFrame({
        'Open':  [100, 100, 100, 90, 100, 100],
        'Close': [100, 100, 100, 100, 100, 100],
    })

    df = features.add_gap_z_signal(data, window=3, k_low=-1.0)

    assert 'signal_gapz' in df.columns
    assert 'score_gapz' in df.columns
    assert df['signal_gapz'].iloc[3]
    assert df['score_gapz'].iloc[3] < -1.0
    assert not df['signal_gapz'].iloc[2]
