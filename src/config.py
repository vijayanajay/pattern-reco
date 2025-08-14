"""
Configuration loading and validation using Pydantic.

This version uses nested, strongly-typed Pydantic models instead of
generic dictionaries. This improves type safety, enables editor autocompletion,
and delegates validation to Pydantic, making the code cleaner and more robust.
"""

from datetime import date
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, root_validator

__all__ = ["load_config", "Config"]


# §1. Nested Configuration Models
# --------------------------------------------------------------------------------------
# Each class represents a section of the YAML configuration file, providing
# validation, default values, and type hints.


class RunConfig(BaseModel):
    """Configuration for a single run."""

    name: str
    t0: date
    seed: int = 42
    output_dir: Path = Path("runs")


class DataConfig(BaseModel):
    """Configuration for data source and processing."""

    source: str = "yfinance"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    start_date: date
    end_date: date
    snapshot_dir: Path
    refresh: bool = False


class UniverseConfig(BaseModel):
    """Configuration for stock universe selection."""

    include_symbols: List[str] = Field(default_factory=list)
    exclude_symbols: List[str] = Field(default_factory=list)
    size: int = 10
    min_turnover: float = 10_000_000.0
    min_price: float = 10.0
    lookback_years: int = 2


class DetectorConfig(BaseModel):
    """Configuration for the signal detector."""

    name: str = "gap_z"
    window_range: List[int] = Field(default_factory=lambda: [20, 60])
    k_low_range: List[float] = Field(default_factory=lambda: [-1.0, -2.0])
    max_hold: int = 22
    min_hit_rate: float = 0.4


class WalkForwardConfig(BaseModel):
    """Configuration for walk-forward validation."""

    is_years: int = 3
    oos_years: int = 1
    holdout_years: int = 2


class SlippageConfig(BaseModel):
    """Configuration for the slippage model."""

    gap_2pct: float = 5.0
    gap_5pct: float = 10.0
    gap_high: float = 20.0


class ExecutionConfig(BaseModel):
    """Configuration for trade execution simulation."""

    circuit_guard_pct: float = 0.10
    fees_bps: float = 10.0
    slippage_model: SlippageConfig = Field(default_factory=SlippageConfig)


class PortfolioConfig(BaseModel):
    """Configuration for portfolio management."""

    max_concurrent: int = 5
    position_size: float = 100_000.0
    equal_weight: bool = True
    reentry_lockout: bool = True


class ReportingConfig(BaseModel):
    """Configuration for output reporting."""

    generate_plots: bool = False
    output_formats: List[Literal["json", "markdown", "csv"]] = Field(
        default_factory=lambda: ["json", "markdown", "csv"]
    )
    include_unfilled: bool = True


# §2. Top-Level Configuration
# --------------------------------------------------------------------------------------


class Config(BaseModel):
    """The root configuration model, composing all nested sections."""

    run: RunConfig
    data: DataConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @root_validator
    def validate_dates(cls, values):
        """Ensures logical consistency between different date fields."""
        run_cfg = values.get("run")
        data_cfg = values.get("data")

        # This check is only performed if both configs are present.
        if run_cfg and data_cfg:
            if data_cfg.end_date <= data_cfg.start_date:
                raise ValueError("data.end_date must be after data.start_date")
            if not (data_cfg.start_date < run_cfg.t0 < data_cfg.end_date):
                raise ValueError("run.t0 must be within the data start and end dates")
        return values

    class Config:
        # Pydantic v1 model configuration
        allow_population_by_field_name = True


# §3. Loading Function
# --------------------------------------------------------------------------------------


# impure
def load_config(config_path: Path) -> Config:
    """
    Loads and validates a YAML configuration file into a Config object.
    #impure: Reads from the filesystem.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        if not isinstance(raw_config, dict):
            raise ValueError("Configuration must be a YAML object.")
        return Config.parse_obj(raw_config)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e
    except (ValidationError, ValueError, TypeError) as e:
        # Re-raise to provide more specific feedback on validation failure.
        raise ValueError(f"Configuration validation failed: {e}") from e
