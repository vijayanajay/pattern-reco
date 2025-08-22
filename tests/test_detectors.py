"""
Tests for signal detection logic.
"""
import pandas as pd
import numpy as np
import pytest

from src.detectors import generate_signals

@pytest.fixture
def sample_gap_data() -> pd.DataFrame:
    """Creates a sample DataFrame with a 'gap_pct' column for testing."""
    # Create a series with a clear outlier
    data = np.random.randn(100) * 0.01  # Normal daily gaps
    data[-5] = -0.10  # A significant negative gap
    return pd.DataFrame({"gap_pct": data})


def test_generate_signals_detects_outlier(sample_gap_data: pd.DataFrame):
    """
    Tests that generate_signals correctly identifies a clear negative outlier.
    """
    window = 20
    k_low = -2.5  # A threshold that the outlier should cross

    signals = generate_signals(sample_gap_data, window=window, k_low=k_low)

    assert isinstance(signals, pd.Series)
    assert signals.dtype == bool
    # The signal should fire on the outlier day
    assert signals.iloc[-5] == True
    # Most other days should not have signals
    assert signals.sum() < 5  # Expect only a few signals, not many


def test_generate_signals_no_signal(sample_gap_data: pd.DataFrame):
    """
    Tests that no signals are generated if the threshold is not met.
    """
    window = 20
    k_low = -5.0  # A very high threshold that should not be crossed

    signals = generate_signals(sample_gap_data, window=window, k_low=k_low)

    assert signals.sum() == 0


def test_generate_signals_bad_input():
    """
    Tests that the function raises errors for invalid input.
    """
    # Test with missing 'gap_pct' column
    with pytest.raises(ValueError, match="must contain a 'gap_pct' column"):
        generate_signals(pd.DataFrame({"close": [1, 2, 3]}), window=10, k_low=-2.0)

    # Test with a positive (invalid) k_low threshold
    with pytest.raises(ValueError, match="k_low threshold must be a negative value"):
        df = pd.DataFrame({"gap_pct": [0.01, -0.01]})
        generate_signals(df, window=1, k_low=2.0)
