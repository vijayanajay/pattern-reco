import json

import numpy as np
import pandas as pd
import pytest

from src.detector import GapZDetector


@pytest.fixture
def sample_detector_data() -> pd.DataFrame:
    """
    Provides data where a specific set of params should be optimal.
    A strong downward gap is included to ensure a signal is generated.
    """
    # Repeat the pattern to have a long enough series for rolling windows
    return pd.DataFrame(
        {
            "Open": [100, 102, 95, 103, 105, 110, 112, 115, 118, 120] * 3,
            "Close": [102, 101, 103, 105, 106, 111, 114, 117, 119, 122] * 3,
        }
    )


@pytest.fixture
def params_grid() -> dict:
    """Provides a sample parameter grid for the detector."""
    return {
        "window": [5, 10],
        "k_low": [-1.0, -1.5],
        "max_hold": [2, 5],
    }


def test_detector_init(params_grid):
    """Tests that the detector is initialized correctly."""
    detector = GapZDetector(params_grid)
    assert detector.params_grid == params_grid
    assert not detector.best_params_  # Should be empty initially


def test_detector_fit_finds_best_params(sample_detector_data, params_grid):
    """Ensures the fit method identifies and stores optimal parameters."""
    detector = GapZDetector(params_grid)
    detector.fit(sample_detector_data)

    assert "window" in detector.best_params_
    assert "k_low" in detector.best_params_
    assert "max_hold" in detector.best_params_
    assert detector.best_params_["window"] in params_grid["window"]
    assert detector.best_params_["k_low"] in params_grid["k_low"]
    assert detector.best_params_["max_hold"] in params_grid["max_hold"]


def test_detector_predict_after_fit(sample_detector_data, params_grid):
    """Tests that predict generates signals after fitting."""
    detector = GapZDetector(params_grid)
    detector.fit(sample_detector_data)
    signals = detector.predict(sample_detector_data)

    assert isinstance(signals, pd.Series)
    assert signals.dtype == bool
    assert len(signals) == len(sample_detector_data)
    assert signals.any()  # Expect at least one signal


def test_detector_predict_raises_error_before_fit(sample_detector_data, params_grid):
    """Verifies that predict cannot be called before fit."""
    detector = GapZDetector(params_grid)
    with pytest.raises(RuntimeError, match="Detector has not been fitted"):
        detector.predict(sample_detector_data)


def test_detector_save_and_load_persists_params(
    tmp_path, sample_detector_data, params_grid
):
    """Checks that parameters are correctly saved to and loaded from a file."""
    # Fit and save detector
    detector1 = GapZDetector(params_grid)
    detector1.fit(sample_detector_data)
    filepath = tmp_path / "test_params.json"
    detector1.save(filepath)
    assert filepath.exists()

    # Load into a new detector instance
    detector2 = GapZDetector(params_grid)
    detector2.load(filepath)

    assert detector1.best_params_ == detector2.best_params_


def test_detector_save_raises_error_before_fit(params_grid, tmp_path):
    """Verifies that save cannot be called before fit."""
    detector = GapZDetector(params_grid)
    with pytest.raises(RuntimeError, match="Nothing to save"):
        detector.save(tmp_path / "params.json")
