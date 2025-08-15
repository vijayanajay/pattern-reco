"""Tests for the vectorbt-based reporting functions."""
import copy
import json
from pathlib import Path
from datetime import date
from typing import Dict, Any, cast
import numpy as np

import pandas as pd
import pytest
import vectorbt as vbt
from rich.console import Console

from src.config import Config, _from_dict
from src.backtest import _run_vbt_backtest
from src.reporting import generate_all_reports

# A complete and valid dictionary for creating a Config object in tests.
FULL_CONFIG_DICT: Dict[str, Any] = {
    "run": {"name": "test_reporting_run", "t0": date(2023, 1, 15), "seed": 42, "output_dir": ""},
    "data": {
        "start_date": date(2023, 1, 1), "end_date": date(2023, 1, 31),
        "source": "yfinance_test", "interval": "1d", "snapshot_dir": "", "refresh": False,
    },
    "universe": {"include_symbols": ["TEST.NS"], "exclude_symbols": [], "size": 1, "min_turnover": 1.0, "min_price": 1.0, "lookback_years": 1},
    "detector": {"name": "gap_z", "window_range": [5], "k_low_range": [-1.0], "max_hold": 5, "min_hit_rate": 0.0},
    "walk_forward": {"is_years": 1, "oos_years": 1, "holdout_years": 1},
    "execution": {"circuit_guard_pct": 0.1, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 1, "gap_5pct": 2, "gap_high": 3}},
    "portfolio": {"max_concurrent": 1, "position_size": 1.0, "equal_weight": True, "reentry_lockout": True},
    "reporting": {"generate_plots": False, "output_formats": ["csv", "json", "markdown"], "include_unfilled": True},
}


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Pytest fixture to create a valid Config dataclass object for testing."""
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["data"]["snapshot_dir"] = str(tmp_path)
    config_dict["run"]["output_dir"] = str(tmp_path)
    return cast(Config, _from_dict(Config, config_dict))


@pytest.fixture
def sample_portfolio(test_config: Config) -> vbt.Portfolio:
    """
    Creates a sample vbt.Portfolio object by running a small backtest.
    This serves as the input for the reporting tests.
    """
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

    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    # Manually insert a very large negative gap to ensure a signal is generated
    df.loc[df.index[50], "gap_pct"] = -0.50
    processed_data = {"TEST.NS": df}

    return _run_vbt_backtest(test_config, processed_data, Console())


def test_generate_all_reports(
    test_config: Config, sample_portfolio: vbt.Portfolio, tmp_path: Path
) -> None:
    """
    Tests that generate_all_reports creates the specified output files.
    """
    run_dir = tmp_path / "test_run_output"
    run_dir.mkdir()

    generate_all_reports(test_config, sample_portfolio, run_dir, Console())

    # Check that the main summary files were created.
    # The trade ledger is only created if there are trades, which we can't
    # guarantee in this synthetic test.
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    # Check the content of the JSON summary
    with (run_dir / "summary.json").open("r") as f:
        summary_data = json.load(f)

    assert summary_data["run_name"] == "test_reporting_run"
    assert "Total Return [%]" in summary_data["metrics"]
    # Check that the metric exists, but don't assert its value, as no
    # trades may be generated in the synthetic test.
    assert "Total Trades" in summary_data["metrics"]
