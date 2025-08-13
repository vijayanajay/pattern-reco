"""
Tests for configuration loading and validation.
"""

import tempfile
from pathlib import Path
from typing import Generator
import yaml

import pytest

from src.config import load_config

# A minimal, valid config for testing
MINIMAL_CONFIG = {
    "run": {"name": "test", "seed": 42},
    "data": {"source": "yfinance", "interval": "1d", "start_date": "2020-01-01", "end_date": "2021-01-01"},
    "universe": {"size": 1, "min_turnover": 0, "min_price": 0, "lookback_years": 1},
    "detector": {"name": "gap_z", "window_range": [10], "k_low_range": [-1.0], "max_hold": 5, "min_hit_rate": 0},
    "walk_forward": {"in_sample_years": 1, "out_sample_years": 1, "holdout_years": 0},
    "execution": {"fees_bps": 10},
    "portfolio": {"max_hold_days": 5, "max_concurrent": 5, "position_size": 10000, "equal_weight": True, "reentry_lockout": True},
    "reporting": {"output_formats": ["json"]},
}

@pytest.fixture
def temp_config_file() -> Generator[Path, None, None]:
    """Pytest fixture to create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(MINIMAL_CONFIG, f)
        config_path = f.name

    yield Path(config_path)

    Path(config_path).unlink()

def test_load_valid_config(temp_config_file: Path) -> None:
    """Test loading a valid configuration file returns a dict."""
    config = load_config(str(temp_config_file))
    assert isinstance(config, dict)
    assert config["run"]["name"] == "test"

def test_load_example_config_file() -> None:
    """Test that the main example config file is valid."""
    config = load_config("config/example.yaml")
    assert isinstance(config, dict)
    assert config["run"]["name"] == "gap_z_example"

def test_missing_config_file() -> None:
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")

def test_invalid_yaml_syntax() -> None:
    """Test error handling for invalid YAML syntax."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("this is: not valid yaml")
        config_path = f.name

    try:
        with pytest.raises(ValueError, match="Configuration validation failed"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()

def test_validation_error_on_invalid_type() -> None:
    """Test that Pydantic validation fails for incorrect types."""
    invalid_config = MINIMAL_CONFIG.copy()
    invalid_config['universe']['size'] = "not-an-integer"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        config_path = f.name

    try:
        with pytest.raises(ValueError, match="Configuration validation failed"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()

def test_date_validation_fails() -> None:
    """Test that validation fails if end_date is before start_date."""
    invalid_config = MINIMAL_CONFIG.copy()
    invalid_config['data']['start_date'] = "2022-01-01"
    invalid_config['data']['end_date'] = "2021-01-01"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config, f)
        config_path = f.name

    try:
        # Pydantic v1 raises a ValueError that contains this message
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()
