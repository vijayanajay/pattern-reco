"""
Configuration loading and validation for the anomaly detection system.
"""

from datetime import date
from pathlib import Path
from typing import Dict, List, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, root_validator, validator

__all__ = ["load_config", "Config"]


class RunConfig(BaseModel):
    name: str
    t0: date
    seed: int = 42
    output_dir: Path = Field(default_factory=lambda: Path("runs"))


class DataConfig(BaseModel):
    source: str = "yfinance"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    start_date: date
    end_date: date
    refresh: bool = False
    snapshot_dir: Path = Field(default_factory=lambda: Path("data/snapshots"))

    @validator("end_date")
    def end_after_start(cls, v: date, values: Dict) -> date:
        if "start_date" in values and v <= values["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v


class UniverseConfig(BaseModel):
    size: int = 10
    min_turnover: float = 1e7
    min_price: float = 10.0
    include_symbols: List[str] = []
    exclude_symbols: List[str] = []
    lookback_years: int = 2


class DetectorConfig(BaseModel):
    name: str = "gap_z"
    window_range: List[int] = [20, 60]
    k_low_range: List[float] = [-1.0, -2.0]
    max_hold: int = 22
    min_hit_rate: float = 0.4


class WalkForwardConfig(BaseModel):
    is_years: int = 3
    oos_years: int = 1
    holdout_years: int = 2


class ExecutionConfig(BaseModel):
    circuit_guard_pct: float = 0.10
    fees_bps: float = 10.0
    slippage_model: Dict[str, float] = {
        "gap_2pct": 5.0,
        "gap_5pct": 10.0,
        "gap_high": 20.0,
    }


class PortfolioConfig(BaseModel):
    max_concurrent: int = 5
    position_size: float = 100000.0
    equal_weight: bool = True
    reentry_lockout: bool = True
    max_hold_days: int = 22


class ReportingConfig(BaseModel):
    generate_plots: bool = False
    output_formats: List[str] = ["json", "markdown", "csv"]
    include_unfilled: bool = True


class Config(BaseModel):
    run: RunConfig
    data: DataConfig
    universe: UniverseConfig
    detector: DetectorConfig
    walk_forward: WalkForwardConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    reporting: ReportingConfig

    @root_validator
    def t0_is_within_data_range(cls, values: Dict) -> Dict:
        run_cfg, data_cfg = values.get("run"), values.get("data")
        if run_cfg and data_cfg:
            if run_cfg.t0 <= data_cfg.start_date:
                raise ValueError("run.t0 must be after data.start_date")
            if run_cfg.t0 >= data_cfg.end_date:
                raise ValueError("run.t0 must be before data.end_date")
        return values


# impure
def load_config(config_path: Path) -> Config:
    """Load and validate YAML configuration file."""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        if not isinstance(raw_config, dict):
            raise ValueError("Configuration must be a YAML object.")
        return Config(**raw_config)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed:\n{e}") from e
