"""Tests for configuration loading and validation."""
import copy
from pathlib import Path
from datetime import date
from typing import Dict, Any

import pytest
import yaml

from src.config import Config, load_config, _from_dict, RunConfig

# A complete and valid dictionary that can be used to construct a Config object.
FULL_CONFIG_DICT: Dict[str, Any] = {
    "run": {"name": "test_run", "t0": "2023-01-15", "seed": 42, "output_dir": "test_output"},
    "data": {
        "source": "yfinance", "interval": "1d", "start_date": "2023-01-01",
        "end_date": "2023-12-31", "snapshot_dir": "test_snapshots", "refresh": False,
    },
    "universe": {
        "include_symbols": ["TEST.NS"], "exclude_symbols": [], "size": 1,
        "min_turnover": 1.0, "min_price": 1.0, "lookback_years": 1,
    },
    "detector": {"name": "gap_z", "window_range": [20], "k_low_range": [-2.0], "max_hold": 10, "min_hit_rate": 0.0},
    "walk_forward": {"is_years": 1, "oos_years": 1, "holdout_years": 1},
    "execution": {"circuit_guard_pct": 0.1, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 1, "gap_5pct": 2, "gap_high": 3}},
    "portfolio": {"max_concurrent": 1, "position_size": 1.0, "equal_weight": True, "reentry_lockout": True},
    "reporting": {"generate_plots": False, "output_formats": ["json"], "include_unfilled": True},
}


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Pytest fixture to create a temporary, valid config file."""
    config_path = tmp_path / "config.yaml"
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    config_dict["data"]["snapshot_dir"] = str(tmp_path)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
    return config_path


def test_load_valid_config(temp_config_file: Path) -> None:
    """Test loading a valid configuration file returns a Config object."""
    config = load_config(temp_config_file)
    assert isinstance(config, Config)
    assert config.run.name == "test_run"


def test_load_example_config_file() -> None:
    """Test that the main example config file is valid."""
    config = load_config(Path("config/example.yaml"))
    assert isinstance(config, Config)


def test_missing_config_file() -> None:
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent.yaml"))


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    """Test error handling for invalid YAML syntax."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("run: { name: test")
    with pytest.raises(ValueError, match="Invalid YAML syntax"):
        load_config(config_path)


def test_date_validation_fails(tmp_path: Path) -> None:
    """Test that validation fails if end_date is before start_date."""
    invalid_config = copy.deepcopy(FULL_CONFIG_DICT)
    invalid_config["data"]["start_date"] = "2022-01-01"
    invalid_config["data"]["end_date"] = "2021-01-01"
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ValueError, match="data.end_date must be after data.start_date"):
        load_config(config_path)


def test_t0_validation_fails(tmp_path: Path) -> None:
    """Test that validation fails if t0 is not within the data date range."""
    invalid_config = copy.deepcopy(FULL_CONFIG_DICT)
    invalid_config["run"]["t0"] = "2024-01-01"  # After end_date
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ValueError, match="run.t0 must be within the data start and end dates"):
        load_config(config_path)


def test_from_dict_conversion() -> None:
    """Tests the internal _from_dict helper for creating nested dataclasses."""
    config_dict = copy.deepcopy(FULL_CONFIG_DICT)
    # The helper expects string dates, so convert them back
    config_dict["run"]["t0"] = str(config_dict["run"]["t0"])
    config_dict["data"]["start_date"] = str(config_dict["data"]["start_date"])
    config_dict["data"]["end_date"] = str(config_dict["data"]["end_date"])

    config = _from_dict(Config, config_dict)
    assert isinstance(config, Config)
    assert isinstance(config.run, RunConfig)
    assert config.run.t0 == date(2023, 1, 15)
    assert config.execution.slippage_model.gap_2pct == 1
