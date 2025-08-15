"""
Tests for CLI interface.
"""
from pathlib import Path
from unittest.mock import MagicMock
import yaml

from typer.testing import CliRunner

from cli import app

# Default CliRunner mixes stderr and stdout into the .output attribute,
# which is what we want for testing console output.
runner = CliRunner()


# Using a full, valid config dictionary to prevent KeyErrors during tests.
FULL_CONFIG_DICT = {
    "run": {"name": "test_cli_run", "t0": "2023-01-15", "seed": 42, "output_dir": ""},
    "data": {
        "source": "yfinance", "interval": "1d", "start_date": "2023-01-01",
        "end_date": "2023-12-31", "snapshot_dir": "", "refresh": False,
    },
    "universe": {
        "include_symbols": ["RELIANCE.NS"], "exclude_symbols": [], "size": 1,
        "min_turnover": 1.0, "min_price": 1.0, "lookback_years": 1,
    },
    "detector": {"name": "gap_z", "window_range": [20], "k_low_range": [-2.0], "max_hold": 10, "min_hit_rate": 0.0},
    "walk_forward": {"is_years": 1, "oos_years": 1, "holdout_years": 1},
    "execution": {"circuit_guard_pct": 0.1, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 1, "gap_5pct": 2, "gap_high": 3}},
    "portfolio": {"max_concurrent": 1, "position_size": 1.0, "equal_weight": True, "reentry_lockout": True},
    "reporting": {"generate_plots": False, "output_formats": ["json"], "include_unfilled": True},
}


def create_temp_config(tmp_path: Path) -> Path:
    """Creates a temporary, valid YAML config file for testing."""
    config_path = tmp_path / "test_config.yaml"
    config_dict = FULL_CONFIG_DICT.copy()
    config_dict["data"]["snapshot_dir"] = str(tmp_path)
    config_dict["run"]["output_dir"] = str(tmp_path)
    config_path.write_text(yaml.dump(config_dict))
    return config_path


def test_cli_help() -> None:
    """Test CLI --help flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Execute the backtest pipeline" in result.output


def test_cli_run_with_missing_config_file() -> None:
    """Test that `run` exits if the config file does not exist."""
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_run_command_runs(mocker, tmp_path: Path) -> None:
    """
    Tests that the `run` command correctly orchestrates the pipeline functions.
    """
    mocker.patch("cli.select_universe", return_value=["RELIANCE.NS"])
    mocker.patch("cli.load_snapshots", return_value={"RELIANCE.NS": MagicMock()})
    mocker.patch("cli.add_features", return_value=MagicMock())
    mocker.patch("cli.run_the_backtest", return_value=MagicMock())
    mocker.patch("cli.generate_all_reports")

    config_path = create_temp_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    assert "Run command finished" in result.output


def test_cli_refresh_command_existing_symbols(mocker, tmp_path: Path) -> None:
    """Tests refresh-data when symbols are discovered in the snapshot dir."""
    m_fetch = mocker.patch("cli.fetch_and_snapshot", return_value=[])
    m_discover = mocker.patch("cli.discover_symbols", return_value=["EXISTING.NS"])
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["refresh-data", "--config", str(config_path)])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    assert "Found 1 existing symbols" in result.output
    m_discover.assert_called_once()
    m_fetch.assert_called_once_with(["EXISTING.NS"], mocker.ANY)


def test_cli_refresh_command_bootstrap_from_config(mocker, tmp_path: Path) -> None:
    """Tests refresh-data when no snapshots exist and it uses the config list."""
    m_fetch = mocker.patch("cli.fetch_and_snapshot", return_value=[])
    m_discover = mocker.patch("cli.discover_symbols", return_value=[])
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["refresh-data", "--config", str(config_path)])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    assert "No existing snapshots found" in result.output
    m_discover.assert_called_once()
    m_fetch.assert_called_once_with(["RELIANCE.NS"], mocker.ANY)
