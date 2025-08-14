import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import Config, RunConfig
from src.reporting import (
    calculate_trade_returns,
    generate_run_manifest,
    generate_summary_json,
    generate_summary_markdown,
    generate_trade_ledger_csv,
)
from src.types import RunContext, RunProposals, RunResult, Trade


from src.config import Config, RunConfig, DataConfig


@pytest.fixture
def mock_run_result() -> RunResult:
    """Provides a realistic RunResult instance for testing."""
    try:
        config = Config(
            run=RunConfig(name="test_run", output_dir="test_output", t0=date(2023, 1, 2)),
            universe={"include_symbols": ["RELIANCE.NS", "TCS.NS"]},
            data=DataConfig(
                snapshot_dir="test_data",
                start_date=date(2023, 1, 1),
                end_date=date(2023, 12, 31),
                source="yfinance",
            ),
            features={},
            detector={},
            trading={},
        )
    except ValidationError as e:
        pytest.fail(f"Failed to create a valid Config for tests: {e}")

    trades = [
        Trade(
            symbol="RELIANCE.NS",
            entry_date=date(2023, 1, 5),
            exit_date=date(2023, 1, 10),
            entry_price=2500.0,
            exit_price=2550.0,
            sample_type="IS",
        ),
        Trade(
            symbol="TCS.NS",
            entry_date=date(2023, 2, 3),
            exit_date=date(2023, 2, 8),
            entry_price=3400.0,
            exit_price=3350.0,
            sample_type="OOS",
        ),
    ]
    signals = pd.DataFrame({"date": [date(2023, 1, 5), date(2023, 2, 3)], "signal": [1, -1]})
    context = RunContext(config=config, git_hash="test_hash", data_hash="test_data_hash")
    proposals = RunProposals(signals=signals, trades=trades)
    metrics = {"total_return_pct": 0.01, "sharpe_ratio": 1.5, "num_trades": 2}
    return RunResult(context=context, proposals=proposals, metrics=metrics)


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory for output files."""
    dir_path = tmp_path / "reporting_output"
    dir_path.mkdir()
    yield dir_path
    shutil.rmtree(dir_path)


def test_calculate_trade_returns(mock_run_result: RunResult):
    """Verifies that trade returns are calculated correctly."""
    returns_df = calculate_trade_returns(mock_run_result.proposals.trades)
    assert not returns_df.empty
    assert len(returns_df) == 2
    assert "return_pct" in returns_df.columns
    assert "duration_days" in returns_df.columns
    # RELIANCE: (2550 / 2500) - 1 = 0.02
    assert returns_df.loc[0, "return_pct"] == pytest.approx(0.02)
    assert returns_df.loc[0, "duration_days"] == 5


def test_generate_trade_ledger_csv(mock_run_result: RunResult, output_dir: Path):
    """Ensures the trade ledger CSV is created with the correct content."""
    generate_trade_ledger_csv(mock_run_result, output_dir)
    ledger_file = output_dir / "trade_ledger.csv"
    assert ledger_file.exists()
    df = pd.read_csv(ledger_file)
    assert len(df) == 2
    assert "return_pct" in df.columns


def test_generate_summary_json(mock_run_result: RunResult, output_dir: Path):
    """Validates the creation and content of the summary JSON file."""
    generate_summary_json(mock_run_result, output_dir)
    summary_file = output_dir / "summary.json"
    assert summary_file.exists()
    with summary_file.open("r") as f:
        data = json.load(f)
    assert data["run_name"] == "test_run"
    assert data["metrics"]["num_trades"] == 2


def test_generate_summary_markdown(mock_run_result: RunResult, output_dir: Path):
    """Checks that the summary Markdown report is generated correctly."""
    generate_summary_markdown(mock_run_result, output_dir)
    md_file = output_dir / "summary.md"
    assert md_file.exists()
    content = md_file.read_text()
    assert "# Run Summary: test_run" in content
    assert "Total Return Pct" in content
    assert "1.5000" in content


def test_generate_run_manifest(mock_run_result: RunResult, output_dir: Path):
    """Verifies the manifest file is created with all required metadata."""
    generate_run_manifest(mock_run_result, output_dir)
    manifest_file = output_dir / "manifest.json"
    assert manifest_file.exists()
    with manifest_file.open("r") as f:
        data = json.load(f)
    assert data["git_hash"] == "test_hash"
    assert data["config"]["run"]["name"] == "test_run"
