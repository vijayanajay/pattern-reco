"""
Tests for stateless detector functions.
"""
from typing import Any, Dict, Generator

import pandas as pd
import pytest
from _pytest.monkeypatch import MonkeyPatch

from src.core import backtest  # Need this for the mocked return calculation
from src.core.detectors import fit_gap_z_detector


@pytest.fixture
def sample_detector_data() -> pd.DataFrame:
    """Provides data with a clear downward gap to ensure a signal."""
    return pd.DataFrame({
        "Open": [100, 102, 95, 103, 105] * 6,
        "Close": [102, 101, 103, 105, 106] * 6,
        "High": [103, 103, 104, 106, 107] * 6,
        "Low": [99, 100, 94, 102, 104] * 6,
        "Volume": [1000] * 30,
    }, index=pd.to_datetime(pd.date_range("2023-01-01", periods=30)))

@pytest.fixture
def params_grid() -> Dict[str, Any]:
    """Provides a sample parameter grid for the detector."""
    return {
        "window": [5, 10],
        "k_low": [-1.0, -2.0],
        "max_hold": [2, 5],
    }

def test_fit_gap_z_detector_returns_best_params(
    sample_detector_data: pd.DataFrame, params_grid: Dict[str, Any], monkeypatch: MonkeyPatch
) -> None:
    """
    Ensures the fit function identifies and returns the optimal parameters
    based on a simple mocked return calculation.
    """
    def mock_run_backtest(
        data: pd.DataFrame, signals: pd.Series, config: Dict[str, Any]
    ) -> pd.DataFrame:
        max_hold = config.get("portfolio", {}).get("max_hold_days", 5)
        if max_hold == 5:
            return pd.DataFrame({"return_pct": [0.1, 0.1]}) # High score
        return pd.DataFrame({"return_pct": [0.01, 0.01]}) # Low score

    monkeypatch.setattr(backtest, "run_backtest", mock_run_backtest)

    best_params = fit_gap_z_detector(sample_detector_data, params_grid, min_hit_rate=0.0)

    assert best_params is not None
    assert isinstance(best_params, dict)
    assert "window" in best_params
    assert "k_low" in best_params
    assert "max_hold" in best_params
    assert best_params["max_hold"] == 5

def test_fit_gap_z_detector_returns_none_if_no_valid_params(
    sample_detector_data: pd.DataFrame, params_grid: Dict[str, Any], monkeypatch: MonkeyPatch
) -> None:
    """
    Ensures the fit function returns None if no parameters meet the criteria (e.g., min_hit_rate).
    """
    def mock_run_backtest(
        data: pd.DataFrame, signals: pd.Series, config: Dict[str, Any]
    ) -> pd.DataFrame:
        return pd.DataFrame({"return_pct": [-0.1, -0.1]})

    monkeypatch.setattr(backtest, "run_backtest", mock_run_backtest)

    best_params = fit_gap_z_detector(sample_detector_data, params_grid, min_hit_rate=0.6)

    assert best_params is None
