"""
Tests for CLI interface.
"""
import tempfile
from pathlib import Path

import yaml
from typer.testing import CliRunner

from cli import app
from src.config import Config

runner = CliRunner()


def create_temp_config(temp_dir: Path, include_symbols: bool = True) -> Path:
    """Creates a temporary YAML config file for testing."""
    test_config = {
        "run": {"name": "test_cli_run"},
        "data": {
            "source": "yfinance_test",
            "interval": "1d",
            "start_date": "2023-01-01",
            "end_date": "2023-01-31",
            "snapshot_dir": str(temp_dir),
        },
        "universe": {
            "min_turnover": 0, "min_price": 0, "size": 1, "lookback_years": 1,
        },
        "detector": {}, "walk_forward": {}, "execution": {}, "portfolio": {}, "reporting": {},
    }
    if include_symbols:
        test_config["universe"]["include_symbols"] = ["RELIANCE.NS"]

    config_path = temp_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)
    return config_path


def test_cli_help() -> None:
    """Test CLI --help flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Execute the backtest pipeline" in result.stdout
    assert "Refresh data snapshots" in result.stdout


def test_cli_run_with_missing_config_file() -> None:
    """Test that `run` exits if the config file does not exist."""
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code == 2
    assert "does not exist" in result.stderr


def test_cli_run_missing_config_arg() -> None:
    """Test that `run` exits if --config is not provided."""
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "Missing option" in result.stderr


def test_cli_refresh_data_missing_config_arg() -> None:
    """Test that `refresh-data` exits if --config is not provided."""
    result = runner.invoke(app, ["refresh-data"])
    assert result.exit_code == 2
    assert "Missing option" in result.stderr


def test_cli_invalid_command() -> None:
    """Test that an invalid command exits."""
    result = runner.invoke(app, ["invalid-command"])
    assert result.exit_code == 2
    assert "No such command" in result.stderr


def test_cli_run_command_runs(mocker) -> None:
    """A simple test to ensure the `run` command executes the pipeline functions."""
    mocker.patch("cli.load_snapshots", return_value=mocker.MagicMock())
    mocker.patch("cli.select_universe", return_value=["TEST.NS"])
    mocker.patch("cli.get_nse_symbols", return_value=["TEST.NS"])

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_temp_config(Path(temp_dir))
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Code simplification review complete" in result.stdout


def test_cli_refresh_command_runs(mocker) -> None:
    """A simple test to ensure the `refresh-data` command executes."""
    mocker.patch("cli.fetch_and_snapshot")
    mocker.patch("cli.get_nse_symbols", return_value=["TEST.NS"])

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_temp_config(Path(temp_dir))
        result = runner.invoke(app, ["refresh-data", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Data refresh completed" in result.stdout

def test_cli_run_missing_symbols_in_config(mocker) -> None:
    """Tests that the CLI exits if include_symbols is missing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_temp_config(Path(temp_dir), include_symbols=False)
        result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "Config must provide a list of symbols" in str(result.exception)
