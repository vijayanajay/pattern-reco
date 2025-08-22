"""
Tests for feature engineering functions.
"""
import pandas as pd
import numpy as np
import pytest

from src.features import add_features


def test_add_features() -> None:
    """
    Test that the add_features function correctly adds all core features
    to a single DataFrame.
    """
    # 1. Prepare a single mock DataFrame
    raw_df = pd.DataFrame({
        "Open": [100.0, 102.0, 101.0],
        "Close": [101.0, 101.0, 103.0],
        "Volume": [1000.0, 1100.0, 1200.0],
    })

    # 2. Run the function
    featured_df = add_features(raw_df)

    # 3. Assert outcomes
    assert isinstance(featured_df, pd.DataFrame)

    # Check for Turnover
    assert "Turnover" in featured_df.columns
    expected_turnover = pd.Series([101000.0, 111100.0, 123600.0])
    pd.testing.assert_series_equal(featured_df["Turnover"], expected_turnover, check_names=False)

    # Check for returns
    assert "returns" in featured_df.columns
    # Day 0: NaN, Day 1: 101/101 - 1 = 0, Day 2: 103/101 - 1 = 0.0198
    assert pd.isna(featured_df["returns"].iloc[0])
    assert featured_df["returns"].iloc[1] == 0.0
    assert featured_df["returns"].iloc[2] == pytest.approx(0.01980198)

    # Check for gap_pct
    assert "gap_pct" in featured_df.columns
    # Day 0: NaN, Day 1: (102 - 101)/101 = 0.0099, Day 2: (101 - 101)/101 = 0.0
    assert pd.isna(featured_df["gap_pct"].iloc[0])
    assert featured_df["gap_pct"].iloc[1] == pytest.approx(0.00990099)
    assert featured_df["gap_pct"].iloc[2] == 0.0

    # Ensure original DataFrame is not modified
    assert "Turnover" not in raw_df.columns
