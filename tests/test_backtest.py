"""Tests for the vectorbt-based backtesting engine."""
import copy
from pathlib import Path
from datetime import date
from typing import Dict, Any, cast
from pytest_mock import MockerFixture
import numpy as np

import pandas as pd
import pytest
import vectorbt as vbt
from rich.console import Console

from src.config import Config, _from_dict
from src.backtest import _run_vbt_backtest, run_pipeline, PipelineError

# A complete and valid dictionary for creating a Config object in tests.
FULL_CONFIG_DICT: Dict[str, Any] = {
    "run": {"name": "test_backtest_run", "t0": date(2023, 1, 15), "seed": 42, "output_dir": ""},
    "data": {
        "start_date": date(2023, 1, 1), "end_date": date(2023, 1, 31),
        "source": "yfinance_test", "interval": "1d", "snapshot_dir": "", "refresh": False,
    },
    "universe": {"include_symbols": ["TEST.NS"], "exclude_symbols": [], "size": 1, "min_turnover": 1.0, "min_price": 1.0, "lookback_years": 1},
    "detector": {"name": "gap_z", "window_range": [5], "k_low_range": [-1.0], "max_hold": 5, "min_hit_rate": 0.0},
    "walk_forward": {"is_years": 1, "oos_years": 1, "holdout_years": 1},
    "execution": {"circuit_guard_pct": 0.1, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 1, "gap_5pct": 2, "gap_high": 3}},
    "portfolio": {"max_concurrent": 1, "position_size": 1.0, "equal_weight": True, "reentry_lockout": True},
    "reporting": {"generate_plots": False, "output_formats": ["json"], "include_unfilled": True},
}


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Pytest fixture to create a valid Config dataclass object for testing."""
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["data"]["snapshot_dir"] = str(tmp_path)
    config_dict["run"]["output_dir"] = str(tmp_path)
    return cast(Config, _from_dict(Config, config_dict))


@pytest.fixture
def processed_data() -> dict[str, pd.DataFrame]:
    """Creates a dictionary of processed data for a single symbol."""
    # Use an even longer series to ensure rolling calculations are stable.
    np.random.seed(42)
    periods = 100
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=periods, freq="D"))
    close_prices = 100 + np.random.randn(periods).cumsum()
    open_prices = close_prices - np.random.uniform(-0.1, 0.1, periods)  # Smaller daily noise

    df = pd.DataFrame({
        "Open": open_prices,
        "Close": close_prices,
        "Volume": np.random.randint(1000, 5000, periods),
    }, index=dates)

    # Add the features that the backtester expects
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    # Manually insert a very large negative gap to ensure a signal is generated
    df.loc[df.index[50], "gap_pct"] = -0.50
    return {"TEST.NS": df}


def test_run_vbt_backtest(
    test_config: Config, processed_data: dict[str, pd.DataFrame]
) -> None:
    """
    Tests that the core `_run_vbt_backtest` function executes and returns a
    valid Portfolio object with trades.
    """
    portfolio = _run_vbt_backtest(test_config, processed_data, Console())

    assert isinstance(portfolio, vbt.Portfolio)
    assert portfolio.stats() is not None
    assert "Total Return [%]" in portfolio.stats()


def test_run_pipeline_success(mocker: MockerFixture, test_config: Config) -> None:
    """Tests the happy path of the main pipeline orchestrator."""
    m_select = mocker.patch("src.backtest.select_universe", return_value=["TEST.NS"])
    m_load = mocker.patch("src.backtest.load_snapshots", return_value={"TEST.NS": "fake_df"})
    m_add_feats = mocker.patch("src.backtest.add_features", return_value="fake_processed_df")
    m_run_vbt = mocker.patch("src.backtest._run_vbt_backtest", return_value="fake_results")
    m_reports = mocker.patch("src.backtest.generate_all_reports")

    run_pipeline(test_config, Console())

    m_select.assert_called_once_with(test_config, mocker.ANY)
    m_load.assert_called_once_with(["TEST.NS"], test_config, mocker.ANY)
    m_add_feats.assert_called_once_with("fake_df")
    m_run_vbt.assert_called_once_with(test_config, {"TEST.NS": "fake_processed_df"}, mocker.ANY)
    m_reports.assert_called_once_with(test_config, "fake_results", mocker.ANY, mocker.ANY)


def test_run_pipeline_empty_universe_raises_error(
    mocker: MockerFixture, test_config: Config
) -> None:
    """Tests that the pipeline exits gracefully if the universe is empty."""
    mocker.patch("src.backtest.select_universe", return_value=[])

    with pytest.raises(PipelineError, match="Universe is empty"):
        run_pipeline(test_config, Console())
