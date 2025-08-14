"""
Tests for the performance metrics module.
"""
from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from src.metrics import aggregate_portfolio_returns
from src.reporting import (
    calculate_benchmark_metrics,
    calculate_oos_is_ratio,
    calculate_per_stock_metrics,
    calculate_trade_returns,
)
from src.types import Trade


@pytest.fixture
def sample_trades() -> List[Trade]:
    """A fixture for a list of sample trades."""
    return [
        Trade(
            symbol="STOCK_A",
            entry_date=date(2023, 1, 5),
            exit_date=date(2023, 1, 12),
            entry_price=100.0,
            exit_price=110.0,
            sample_type="IS",
        ),
        Trade(
            symbol="STOCK_A",
            entry_date=date(2023, 1, 15),
            exit_date=date(2023, 1, 20),
            entry_price=115.0,
            exit_price=105.0,
            sample_type="IS",
        ),
        Trade(
            symbol="STOCK_B",
            entry_date=date(2023, 1, 8),
            exit_date=date(2023, 1, 18),
            entry_price=200.0,
            exit_price=220.0,
            sample_type="OOS",
        ),
    ]


@pytest.fixture
def daily_prices(sample_trades: List[Trade]) -> Dict[str, pd.Series]:
    """
    Fixture for daily prices, ensuring prices match trade entry/exit points,
    and includes a gap to test forward-filling.
    """
    all_dates = set()
    for trade in sample_trades:
        all_dates.add(trade.entry_date)
        all_dates.add(trade.exit_date)

    min_date = date(2023, 1, 1)
    max_date = date(2023, 1, 31)

    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    prices = {}
    for symbol in set(t.symbol for t in sample_trades):
        # Create a dummy price series
        price_series = pd.Series(np.linspace(100, 150, len(date_range)), index=date_range)

        # Overwrite with actual trade prices to ensure consistency
        for trade in sample_trades:
            if trade.symbol == symbol:
                price_series[pd.to_datetime(trade.entry_date)] = trade.entry_price
                price_series[pd.to_datetime(trade.exit_date)] = trade.exit_price

        # Introduce a gap to test ffill
        if symbol == "STOCK_A":
            price_series.loc[pd.to_datetime("2023-01-09")] = np.nan

        prices[symbol] = price_series

    return prices


def test_calculate_trade_returns(sample_trades: List[Trade]):
    """Test the calculation of returns from a list of trades."""
    returns_df = calculate_trade_returns(sample_trades)
    assert isinstance(returns_df, pd.DataFrame)
    assert len(returns_df) == 3
    assert "return_pct" in returns_df.columns
    assert "duration_days" in returns_df.columns

    expected_returns = [0.10, -0.0869565, 0.10]
    expected_durations = [7, 5, 10]

    for i, trade in enumerate(sample_trades):
        assert returns_df.iloc[i]["symbol"] == trade.symbol
        assert pytest.approx(returns_df.iloc[i]["return_pct"]) == expected_returns[i]
        assert returns_df.iloc[i]["duration_days"] == expected_durations[i]


def test_calculate_trade_returns_empty():
    """Test that calculating returns on no trades returns an empty DataFrame."""
    assert calculate_trade_returns([]).empty


def test_calculate_per_stock_metrics(sample_trades: List[Trade]):
    """Test the calculation of metrics per stock."""
    returns_df = calculate_trade_returns(sample_trades)
    metrics_df = calculate_per_stock_metrics(returns_df)

    assert isinstance(metrics_df, pd.DataFrame)
    assert "trade_count" in metrics_df.columns
    assert "median_return_pct" in metrics_df.columns
    assert "p5_return_pct" in metrics_df.columns
    assert "hit_rate" in metrics_df.columns

    # STOCK_A metrics
    a_metrics = metrics_df.loc["STOCK_A"]
    assert a_metrics["trade_count"] == 2
    assert pytest.approx(a_metrics["median_return_pct"]) == 0.006521739
    assert pytest.approx(a_metrics["p5_return_pct"]) == -0.07760865
    assert pytest.approx(a_metrics["hit_rate"]) == 0.5

    # STOCK_B metrics
    b_metrics = metrics_df.loc["STOCK_B"]
    assert b_metrics["trade_count"] == 1
    assert pytest.approx(b_metrics["median_return_pct"]) == 0.10
    assert pytest.approx(b_metrics["hit_rate"]) == 1.0


def test_calculate_per_stock_metrics_empty():
    """Test that per-stock metrics on an empty DataFrame returns an empty DataFrame."""
    assert calculate_per_stock_metrics(pd.DataFrame()).empty


def test_calculate_benchmark_metrics():
    """Test benchmark metrics calculation."""
    prices = pd.Series([100, 110, 121], index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]))
    metrics = calculate_benchmark_metrics(prices)
    assert isinstance(metrics, dict)
    assert "total_return_pct" in metrics
    assert pytest.approx(metrics["total_return_pct"]) == 0.21


def test_calculate_benchmark_metrics_empty():
    """Test that benchmark metrics on empty data returns zero."""
    metrics = calculate_benchmark_metrics(pd.Series(dtype=float))
    assert metrics["total_return_pct"] == 0.0


def test_aggregate_portfolio_returns(sample_trades: List[Trade], daily_prices: Dict[str, pd.Series]):
    """Test the aggregation of portfolio returns over time."""
    start_date = date(2023, 1, 1)
    end_date = date(2023, 1, 31)

    portfolio_value = aggregate_portfolio_returns(
        trades=sample_trades,
        daily_prices=daily_prices,
        initial_capital=1_000_000,
        position_size=100_000,
        start_date=start_date,
        end_date=end_date,
    )

    assert isinstance(portfolio_value, pd.Series)
    assert portfolio_value.index.is_monotonic_increasing
    assert portfolio_value.index.min() == pd.to_datetime(start_date)
    assert portfolio_value.index.max() == pd.to_datetime(end_date)

    # Initial capital before any trades
    assert portfolio_value.loc[pd.to_datetime("2023-01-04")] == 1_000_000

    # Jan 5: Enter first trade (STOCK_A). Price is 100.
    # Holdings value = 100k * (100/100) = 100k. Cash = 900k. Total = 1M.
    assert pytest.approx(portfolio_value.loc[pd.to_datetime("2023-01-05")]) == 1_000_000

    price_a_jan_9_ffill = daily_prices["STOCK_A"].ffill().loc[pd.to_datetime("2023-01-09")]
    val_a = 100_000 * (price_a_jan_9_ffill / 100.0)
    price_b_jan_9 = daily_prices["STOCK_B"].loc[pd.to_datetime("2023-01-09")]
    val_b = 100_000 * (price_b_jan_9 / 200.0)
    expected_val_jan_9 = 800_000 + val_a + val_b
    assert pytest.approx(portfolio_value.loc[pd.to_datetime("2023-01-09")]) == expected_val_jan_9

    final_cash = 1_000_000 + 10_000 - 8695.65217 + 10_000
    assert pytest.approx(portfolio_value.iloc[-1], abs=1) == final_cash


def test_calculate_oos_is_ratio(sample_trades: List[Trade]):
    """Test the OOS/IS performance ratio calculation."""
    trade_returns = calculate_trade_returns(sample_trades)

    # In sample_trades:
    # IS returns: 0.10, -0.087. Median = 0.0065
    # OOS returns: 0.10. Median = 0.10
    # Ratio = 0.10 / 0.0065 = 15.33
    ratio = calculate_oos_is_ratio(trade_returns)
    assert pytest.approx(ratio) == 0.10 / 0.006521739

    # Test with no OOS trades
    is_only_returns = trade_returns[trade_returns["sample_type"] == "IS"]
    assert calculate_oos_is_ratio(is_only_returns) == 0.0

    # Test with no IS trades
    oos_only_returns = trade_returns[trade_returns["sample_type"] == "OOS"]
    assert calculate_oos_is_ratio(oos_only_returns) == 0.0

    # Test with negative IS performance
    trade_returns.loc[trade_returns["sample_type"] == "IS", "return_pct"] = -0.1
    assert calculate_oos_is_ratio(trade_returns) == 0.0

    # Test with empty dataframe
    assert calculate_oos_is_ratio(pd.DataFrame()) == 0.0
