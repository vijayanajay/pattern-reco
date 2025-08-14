"""
Tests for CLI interface.
"""
from pathlib import Path

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def create_temp_config(tmp_path: Path) -> Path:
    """Creates a temporary YAML config file for testing."""
    config_path = tmp_path / "test_config.yaml"
    # Note: Using a raw string to avoid a dependency on pyyaml in the test suite.
    config_content = f"""
run:
  name: test_cli_run
  t0: "2023-01-01"
data:
  source: yfinance_test
  interval: 1d
  start_date: "2022-01-01"
  end_date: "2023-01-31"
  snapshot_dir: "{tmp_path}"
universe:
  include_symbols: ["RELIANCE.NS"]
# These sections are empty as they are not used in the CLI tests.
detector: {{}}
walk_forward: {{}}
execution: {{}}
portfolio: {{}}
reporting: {{}}
"""
    config_path.write_text(config_content)
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
    # Typer exits with 2 for file validation errors
    assert result.exit_code == 2
    # The error message should be in the combined output stream.
    assert "does not exist" in result.output


def test_cli_run_command_runs(tmp_path: Path) -> None:
    """A simple test to ensure the `run` command executes."""
    config_path = create_temp_config(tmp_path)
    result = runner.invoke(app, ["run", "--config", str(config_path)])

    assert result.exit_code == 0
    # The pipeline now generates mock results and reports.
    # This output is sufficient to confirm the command ran.
    assert "Backtest started for run" in result.output


def test_cli_refresh_command_existing_symbols(mocker, tmp_path: Path) -> None:
    """Tests refresh-data when symbols are discovered in the snapshot dir."""
    m_fetch = mocker.patch("cli.fetch_and_snapshot", return_value=[])
    m_discover = mocker.patch("cli.discover_symbols", return_value=["EXISTING.NS"])
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["refresh-data", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "Found 1 existing symbols. Refreshing them." in result.output
    m_discover.assert_called_once()
    m_fetch.assert_called_once_with(["EXISTING.NS"], mocker.ANY)


def test_cli_refresh_command_bootstrap_from_config(mocker, tmp_path: Path) -> None:
    """Tests refresh-data when no snapshots exist and it uses the config list."""
    m_fetch = mocker.patch("cli.fetch_and_snapshot", return_value=[])
    m_discover = mocker.patch("cli.discover_symbols", return_value=[])
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["refresh-data", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "No existing snapshots found" in result.output
    m_discover.assert_called_once()
    m_fetch.assert_called_once_with(["RELIANCE.NS"], mocker.ANY)
