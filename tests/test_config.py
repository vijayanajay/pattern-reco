"""
Tests for configuration loading and validation.
"""

import pytest
import tempfile
from pathlib import Path

from src.config import load_config, create_example_config


def test_load_valid_config():
    """Test loading a valid configuration file."""
    example_config = create_example_config()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(example_config, f)
        config_path = f.name
    
    try:
        config = load_config(config_path)
        assert isinstance(config, dict)
        assert config["run"]["name"] == "gap_z_example"
        assert config["data"]["interval"] == "1d"
        assert config["universe"]["size"] == 10
    finally:
        Path(config_path).unlink()


def test_missing_config_file():
    """Test error handling for missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_invalid_yaml():
    """Test error handling for invalid YAML syntax."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: syntax:")
        config_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Invalid YAML syntax"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_config_validation():
    """Test configuration validation."""
    # Test invalid interval
    invalid_config = create_example_config()
    invalid_config['data']['interval'] = 'invalid'
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(invalid_config, f)
        config_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Configuration validation failed"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_date_validation():
    """Test date order validation."""
    invalid_config = create_example_config()
    invalid_config['data']['start_date'] = "2025-01-01"
    invalid_config['data']['end_date'] = "2020-01-01"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(invalid_config, f)
        config_path = f.name
    
    try:
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            load_config(config_path)
    finally:
        Path(config_path).unlink()


def test_example_config_file():
    """Test that the example config file is valid."""
    config = load_config("config/example.yaml")
    assert isinstance(config, dict)
    assert config["run"]["name"] == "gap_z_example"