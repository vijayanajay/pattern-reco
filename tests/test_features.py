"""
Tests for feature engineering functions.
"""
import pandas as pd

from src.features import add_features


def test_add_features() -> None:
    """
    Test that the add_features function correctly adds the Turnover column.
    """
    # 1. Prepare mock data
    raw_data = {
        "STOCK_A": pd.DataFrame({"Close": [100.0, 101.0], "Volume": [1000.0, 1100.0]}),
        "STOCK_B": pd.DataFrame({"Close": [50.0, 50.5], "Volume": [5000.0, 5500.0]}),
    }

    # 2. Run the function
    featured_data = add_features(raw_data)

    # 3. Assert outcomes
    assert "STOCK_A" in featured_data
    df_a = featured_data["STOCK_A"]
    assert "Turnover" in df_a.columns
    assert df_a["Turnover"].iloc[0] == 100.0 * 1000.0
    assert df_a["Turnover"].iloc[1] == 101.0 * 1100.0

    assert "STOCK_B" in featured_data
    df_b = featured_data["STOCK_B"]
    assert "Turnover" in df_b.columns
    assert df_b["Turnover"].iloc[0] == 50.0 * 5000.0
    assert df_b["Turnover"].iloc[1] == 50.5 * 5500.0
