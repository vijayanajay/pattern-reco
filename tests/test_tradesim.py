import pandas as pd
import numpy as np
import pytest

from src.tradesim import calculate_returns


@pytest.fixture
def sample_trade_data() -> pd.DataFrame:
    """Provides a standard OHLC DataFrame for trade simulation tests."""
    return pd.DataFrame(
        {
            "Open": [100, 102, 101, 103, 105, 110, 112, 115, 118, 120],
            "Close": [102, 101, 103, 105, 106, 111, 114, 117, 119, 122],
        }
    )


def test_calculate_returns_normal_case(sample_trade_data):
    """Tests a typical scenario with multiple successful trades."""
    signals = pd.Series([False, True, False, False, True, False, False, False, False, False])
    max_hold = 2
    returns = calculate_returns(sample_trade_data, signals, max_hold)

    # Expected returns:
    # Trade 1 (signal @1): Enter @2 (101), Exit @4 (106) -> (106-101)/101 = 0.0495
    # Trade 2 (signal @4): Enter @5 (110), Exit @7 (117) -> (117-110)/110 = 0.0636
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], 0.04950495)
    assert np.isclose(returns.iloc[1], 0.06363636)


def test_calculate_returns_no_signals(sample_trade_data):
    """Ensures no returns are calculated when there are no signals."""
    signals = pd.Series([False] * len(sample_trade_data))
    returns = calculate_returns(sample_trade_data, signals, max_hold=5)
    assert returns.empty


def test_calculate_returns_trade_goes_out_of_bounds(sample_trade_data):
    """Tests that signals too close to the end of the data are ignored."""
    # Signal at index 8 -> entry at 9, exit at 9 + 5 = 14 (out of bounds)
    signals = pd.Series([False] * 8 + [True, False])
    returns = calculate_returns(sample_trade_data, signals, max_hold=5)
    assert returns.empty


def test_calculate_returns_with_nan_in_prices(sample_trade_data):
    """Ensures trades with NaN prices are skipped."""
    data = sample_trade_data.copy()
    data.loc[5, "Open"] = np.nan  # Invalidate the second trade's entry price

    signals = pd.Series([False, True, False, False, True, False, False, False, False, False])
    returns = calculate_returns(data, signals, max_hold=2)

    # Only the first trade should be valid
    assert len(returns) == 1
    assert np.isclose(returns.iloc[0], 0.04950495)


def test_calculate_returns_empty_input():
    """Tests behavior with empty DataFrame and Series."""
    data = pd.DataFrame({"Open": [], "Close": []})
    signals = pd.Series([], dtype=bool)
    returns = calculate_returns(data, signals, max_hold=5)
    assert returns.empty


def test_calculate_returns_input_validation():
    """Verifies that input validation raises appropriate errors."""
    df = pd.DataFrame({"A": [1]})
    s = pd.Series([True])
    with pytest.raises(ValueError, match="must contain 'Open' and 'Close'"):
        calculate_returns(df, s, 5)
