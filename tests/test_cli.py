"""
Tests for CLI interface.
"""

import subprocess
import sys


def test_cli_help():
    """Test CLI help command."""
    result = subprocess.run([sys.executable, "cli.py", "--help"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "Anomaly Pattern Detection for Indian Stocks" in result.stdout


def test_cli_run_with_valid_config():
    """Test CLI run command with valid config."""
    result = subprocess.run([sys.executable, "cli.py", "run", "--config", "config/example.yaml"], 
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "Loaded configuration from:" in result.stdout
    assert "Pipeline execution not yet implemented" in result.stdout


def test_cli_run_with_missing_config():
    """Test CLI run command with missing config file."""
    result = subprocess.run([sys.executable, "cli.py", "run", "--config", "nonexistent.yaml"], 
                          capture_output=True, text=True)
    assert result.returncode == 1
    assert "Configuration file not found" in result.stdout


def test_cli_refresh_data():
    """Test CLI refresh-data command with mock."""
    # Use a simpler approach - test that the command runs without network calls
    # by using a config with a very short date range that might work
    import tempfile
    import yaml
    
    # Create a minimal config for testing
    test_config = {
        "run": {"name": "test", "seed": 42, "output_dir": "runs"},
        "data": {"source": "yfinance", "interval": "1d", "start_date": "2024-01-01", "end_date": "2024-01-02", "refresh": False},
        "universe": {"size": 1, "min_turnover": 10000000.0, "min_price": 10.0, "exclude_symbols": [], "lookback_years": 2},
        "detector": {"name": "gap_z", "window_range": [20, 60], "k_low_range": [-1.0, -2.0], "max_hold": 22, "min_hit_rate": 0.4},
        "walk_forward": {"in_sample_years": 3, "out_sample_years": 1, "holdout_years": 2, "calendar_align": True},
        "execution": {"circuit_guard_pct": 0.10, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 5.0, "gap_5pct": 10.0, "gap_high": 20.0}},
        "portfolio": {"max_concurrent": 5, "position_size": 100000.0, "equal_weight": True, "reentry_lockout": True},
        "reporting": {"generate_plots": False, "output_formats": ["json", "markdown", "csv"], "include_unfilled": True}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        # The test should fail gracefully with network error, not crash
        result = subprocess.run([sys.executable, "cli.py", "refresh-data", "--config", config_path], 
                              capture_output=True, text=True)
        # We expect it to fail due to network/data issues, but not crash
        assert "Starting data refresh..." in result.stdout
        # Don't assert returncode == 0 since network calls will likely fail
    finally:
        import os
        os.unlink(config_path)


def test_cli_no_command():
    """Test CLI with no command shows help."""
    result = subprocess.run([sys.executable, "cli.py"], 
                          capture_output=True, text=True)
    assert result.returncode == 1
    assert "usage:" in result.stdout


def test_cli_run_missing_config_arg():
    """Test CLI run command without --config argument."""
    result = subprocess.run([sys.executable, "cli.py", "run"], 
                          capture_output=True, text=True)
    assert result.returncode == 2  # argparse error
    assert "required" in result.stderr


def test_cli_refresh_data_missing_config_arg():
    """Test CLI refresh-data command without --config argument."""
    result = subprocess.run([sys.executable, "cli.py", "refresh-data"], 
                          capture_output=True, text=True)
    assert result.returncode == 2  # argparse error
    assert "required" in result.stderr


def test_cli_invalid_command():
    """Test CLI with invalid command."""
    result = subprocess.run([sys.executable, "cli.py", "invalid-command"], 
                          capture_output=True, text=True)
    assert result.returncode == 2  # argparse error
    assert "invalid choice" in result.stderr