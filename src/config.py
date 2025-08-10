"""
Configuration loading and validation for the anomaly detection system.
Single YAML file controls all aspects of the pipeline.
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Literal

import yaml
from pydantic import BaseModel, validator

__all__ = ["load_config", "create_example_config"]


class RunConfig(BaseModel):
    name: str
    seed: int = 42
    output_dir: str = "runs"


class DataConfig(BaseModel):
    source: str = "yfinance"
    interval: Literal["1d", "1wk", "1mo"] = "1d"
    start_date: date
    end_date: date
    refresh: bool = False
    
    @validator('end_date')
    def end_after_start(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class UniverseConfig(BaseModel):
    size: int = 10
    min_turnover: float = 10000000.0
    min_price: float = 10.0
    exclude_symbols: List[str] = []
    lookback_years: int = 2


class DetectorConfig(BaseModel):
    name: str = "gap_z"
    window_range: List[int] = [20, 60]
    k_low_range: List[float] = [-1.0, -2.0]
    max_hold: int = 22
    min_hit_rate: float = 0.4


class WalkForwardConfig(BaseModel):
    in_sample_years: int = 3
    out_sample_years: int = 1
    holdout_years: int = 2
    calendar_align: bool = True


class ExecutionConfig(BaseModel):
    circuit_guard_pct: float = 0.10
    fees_bps: float = 10.0
    slippage_model: Dict[str, float] = {
        "gap_2pct": 5.0,
        "gap_5pct": 10.0,
        "gap_high": 20.0
    }


class PortfolioConfig(BaseModel):
    max_concurrent: int = 5
    position_size: float = 100000.0
    equal_weight: bool = True
    reentry_lockout: bool = True


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


def load_config(config_path: str) -> Dict[str, Any]:  # impure
    """
    Load and validate YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If YAML parsing fails or validation fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with config_file.open('r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}")
    
    if not isinstance(raw_config, dict):
        raise ValueError(f"Configuration must be a YAML object, got {type(raw_config)}")
    
    # Validate with Pydantic
    try:
        config_model = Config(**raw_config)
        return config_model.dict()
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration dictionary."""
    return {
        "run": {
            "name": "gap_z_example",
            "seed": 42,
            "output_dir": "runs"
        },
        "data": {
            "source": "yfinance",
            "interval": "1d",
            "start_date": "2010-01-01",
            "end_date": "2025-01-01",
            "refresh": False
        },
        "universe": {
            "size": 10,
            "min_turnover": 10000000.0,
            "min_price": 10.0,
            "exclude_symbols": [],
            "lookback_years": 2
        },
        "detector": {
            "name": "gap_z",
            "window_range": [20, 60],
            "k_low_range": [-1.0, -2.0],
            "max_hold": 22,
            "min_hit_rate": 0.4
        },
        "walk_forward": {
            "in_sample_years": 3,
            "out_sample_years": 1,
            "holdout_years": 2,
            "calendar_align": True
        },
        "execution": {
            "circuit_guard_pct": 0.10,
            "fees_bps": 10.0,
            "slippage_model": {
                "gap_2pct": 5.0,
                "gap_5pct": 10.0,
                "gap_high": 20.0
            }
        },
        "portfolio": {
            "max_concurrent": 5,
            "position_size": 100000.0,
            "equal_weight": True,
            "reentry_lockout": True
        },
        "reporting": {
            "generate_plots": False,
            "output_formats": ["json", "markdown", "csv"],
            "include_unfilled": True
        }
    }