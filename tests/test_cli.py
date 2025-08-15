"""Tests for the CLI interface."""
from pathlib import Path
import yaml
from typer.testing import CliRunner
from typing import Dict, Any
from pytest_mock import MockerFixture

from cli import app
from src.backtest import PipelineError

runner = CliRunner()


# A valid config is needed for the CLI to even attempt to run the pipeline.
# The orchestrator functions are mocked, so the content doesn't have to be perfect,
# but it must pass the initial validation in `load_config`.
def create_temp_config(tmp_path: Path) -> Path:
    """Creates a temporary, valid YAML config file for testing."""
    config_path = tmp_path / "test_config.yaml"
    config_dict: Dict[str, Any] = {
        "run": {"name": "test", "t0": "2023-01-15", "seed": 42, "output_dir": str(tmp_path)},
        "data": {
            "source": "test", "interval": "1d", "start_date": "2023-01-01",
            "end_date": "2023-12-31", "snapshot_dir": str(tmp_path), "refresh": False,
        },
        "universe": {"include_symbols": ["TEST.NS"], "exclude_symbols": [], "size": 1, "min_turnover": 1.0, "min_price": 1.0, "lookback_years": 1},
        "detector": {"name": "test", "window_range": [1], "k_low_range": [-1.0], "max_hold": 1, "min_hit_rate": 0.0},
        "walk_forward": {"is_years": 1, "oos_years": 1, "holdout_years": 1},
        "execution": {"circuit_guard_pct": 0.1, "fees_bps": 10.0, "slippage_model": {"gap_2pct": 1, "gap_5pct": 2, "gap_high": 3}},
        "portfolio": {"max_concurrent": 1, "position_size": 1.0, "equal_weight": True, "reentry_lockout": True},
        "reporting": {"generate_plots": False, "output_formats": ["json"], "include_unfilled": True},
    }
    config_path.write_text(yaml.dump(config_dict))
    return config_path


def test_cli_run_success(mocker: MockerFixture, tmp_path: Path) -> None:
    """Tests that the `run` command calls the pipeline orchestrator."""
    mock_run_pipeline = mocker.patch("cli.run_pipeline")
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["run", "-c", str(config_path)])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    assert "Backtest run finished" in result.output
    mock_run_pipeline.assert_called_once()


def test_cli_run_pipeline_error(mocker: MockerFixture, tmp_path: Path) -> None:
    """Tests that `run` handles a PipelineError gracefully."""
    mocker.patch("cli.run_pipeline", side_effect=PipelineError("Test error"))
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["run", "-c", str(config_path)])

    assert result.exit_code == 1
    assert "Pipeline Error: Test error" in result.output


def test_cli_refresh_success(mocker: MockerFixture, tmp_path: Path) -> None:
    """Tests that the `refresh-data` command calls the refresh orchestrator."""
    mock_refresh_data = mocker.patch("cli.refresh_market_data")
    config_path = create_temp_config(tmp_path)

    result = runner.invoke(app, ["refresh-data", "-c", str(config_path)])

    assert result.exit_code == 0, f"CLI exited with error: {result.output}"
    mock_refresh_data.assert_called_once()


def test_cli_help() -> None:
    """Test CLI --help flag."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Execute the backtest pipeline" in result.output


def test_cli_missing_config() -> None:
    """Test that commands exit if the config file does not exist."""
    result = runner.invoke(app, ["run", "--config", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "does not exist" in result.output
