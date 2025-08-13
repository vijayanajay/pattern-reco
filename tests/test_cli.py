"""
Tests for CLI interface.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


# Helper to create a temporary config file
def create_temp_config() -> Path:
    """Creates a temporary YAML config file for testing."""
    test_config = {
        "run": {"name": "test", "seed": 42},
        "data": {
            "source": "yfinance",
            "interval": "1d",
            "start_date": "2024-01-01",
            "end_date": "2024-01-03",
            "snapshot_dir": "tests/temp_data/snapshots",
        },
        "universe": {"size": 1, "min_turnover": 0, "min_price": 0, "lookback_years": 1},
        "detector": {"name": "gap_z", "window_range": [10], "k_low_range": [-1.0], "max_hold": 5, "min_hit_rate": 0},
        "walk_forward": {"in_sample_years": 1, "out_sample_years": 1, "holdout_years": 0},
        "execution": {"fees_bps": 10},
        "portfolio": {"max_hold_days": 5, "max_concurrent": 5, "position_size": 10000, "equal_weight": True, "reentry_lockout": True},
        "reporting": {"output_formats": ["json"]},
    }
    # Use a temporary directory for the config file
    temp_dir = tempfile.mkdtemp()
    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    return config_path

def test_cli_help() -> None:
    """Test CLI help command."""
    result = subprocess.run([sys.executable, "cli.py", "--help"],
                          capture_output=True, text=True)
    assert result.returncode == 0
    assert "A stock pattern detection system." in result.stdout

def test_cli_run_and_refresh() -> None:
    """Test CLI run and refresh-data commands together."""
    config_path = create_temp_config()
    snapshot_dir = Path("tests/temp_data/snapshots/yfinance_1d")

    try:
        # 1. Run refresh-data first. It will fail to download but should create dirs.
        result_refresh = subprocess.run(
            [sys.executable, "cli.py", "refresh-data", "--config", str(config_path)],
            capture_output=True, text=True
        )
        assert result_refresh.returncode == 0
        assert snapshot_dir.is_dir()

        # 2. Create a fake snapshot file to allow universe selection to pass.
        import pandas as pd
        fake_data = pd.DataFrame({
            'Open': [100, 101], 'High': [102, 102], 'Low': [99, 100],
            'Close': [101, 101], 'Volume': [100000, 120000]
        }, index=pd.to_datetime(['2019-12-30', '2019-12-31']))

        # One of the symbols from get_nse_symbols()
        fake_snapshot_path = snapshot_dir / "RELIANCE.NS.parquet"
        fake_data.to_parquet(fake_snapshot_path)

        # 3. Now run the pipeline.
        result_run = subprocess.run(
            [sys.executable, "cli.py", "run", "--config", str(config_path)],
            capture_output=True, text=True
        )
        assert result_run.returncode == 0
        assert "Pipeline run finished" in result_run.stdout

    finally:
        import shutil
        if config_path.parent.exists():
            shutil.rmtree(config_path.parent)
        if snapshot_dir.parent.exists():
            shutil.rmtree(snapshot_dir.parent)

def test_cli_run_with_missing_config_file() -> None:
    """Test CLI run command with missing config file."""
    result = subprocess.run([sys.executable, "cli.py", "run", "--config", "nonexistent.yaml"],
                          capture_output=True, text=True)
    assert result.returncode == 1
    assert "Configuration file not found" in result.stdout

def test_cli_no_command() -> None:
    """Test CLI with no command shows help."""
    result = subprocess.run([sys.executable, "cli.py"],
                          capture_output=True, text=True)
    assert result.returncode == 1
    assert "usage:" in result.stdout

def test_cli_run_missing_config_arg() -> None:
    """Test CLI run command without --config argument."""
    result = subprocess.run([sys.executable, "cli.py", "run"],
                          capture_output=True, text=True)
    assert result.returncode == 2  # argparse error
    assert "required" in result.stderr

def test_cli_refresh_data_missing_config_arg() -> None:
    """Test CLI refresh-data command without --config argument."""
    result = subprocess.run([sys.executable, "cli.py", "refresh-data"],
                          capture_output=True, text=True)
    assert result.returncode == 2  # argparse error
    assert "required" in result.stderr

def test_cli_invalid_command() -> None:
    """Test CLI with invalid command."""
    result = subprocess.run([sys.executable, "cli.py", "invalid-command"],
                          capture_output=True, text=True)
    assert result.returncode == 2  # argparse error
    assert "invalid choice" in result.stderr
