"""
Tests for configuration loading and validation.
"""
import copy
from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config

def minimal_config(tmp_path: Path) -> dict:
    """Factory to create a minimal, valid config for testing."""
    return {
        "run": {"name": "test", "t0": "2020-06-01"},
        "data": {
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
            "snapshot_dir": str(tmp_path),
        },
        "universe": {}, "detector": {}, "walk_forward": {},
        "execution": {}, "portfolio": {}, "reporting": {},
    }


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Pytest fixture to create a temporary config file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(minimal_config(tmp_path), f)
    return config_path


def test_load_valid_config(temp_config_file: Path) -> None:
    """Test loading a valid configuration file returns a Config object."""
    config = load_config(temp_config_file)
    assert isinstance(config, Config)
    assert config.run.name == "test"
    assert config.data.snapshot_dir == temp_config_file.parent


def test_load_example_config_file() -> None:
    """Test that the main example config file is valid."""
    config = load_config(Path("config/example.yaml"))
    assert isinstance(config, Config)
    assert config.run.name == "gap_z_example"


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
    invalid_config = minimal_config(tmp_path)
    invalid_config["data"]["start_date"] = "2022-01-01"
    invalid_config["data"]["end_date"] = "2021-01-01"
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ValueError, match="data.end_date must be after data.start_date"):
        load_config(config_path)


def test_t0_validation_fails(tmp_path: Path) -> None:
    """Test that validation fails if t0 is not within the data date range."""
    invalid_config = minimal_config(tmp_path)
    invalid_config["run"]["t0"] = "2019-12-31"  # Before start_date
    config_path = tmp_path / "invalid.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(invalid_config, f)

    with pytest.raises(ValueError, match="run.t0 must be within the data start and end dates"):
        load_config(config_path)
