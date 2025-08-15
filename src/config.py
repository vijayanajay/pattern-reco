"""
Configuration loading and validation for the pattern-reco application.

This module uses standard library dataclasses for configuration objects.
It avoids external dependencies like Pydantic, favoring explicit, pure
validation functions. This makes the configuration process transparent
and easy to debug.
"""

import yaml
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Literal, Dict, Any, Type, cast

__all__ = ["load_config", "Config"]


# §1. Nested Configuration Dataclasses
# --------------------------------------------------------------------------------------
# Simple, frozen dataclasses replace Pydantic models for clarity and performance.


@dataclass(frozen=True)
class RunConfig:
    name: str
    t0: date
    seed: int
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    source: str
    interval: Literal["1d", "1wk", "1mo"]
    start_date: date
    end_date: date
    snapshot_dir: Path
    refresh: bool


@dataclass(frozen=True)
class UniverseConfig:
    include_symbols: List[str]
    exclude_symbols: List[str]
    size: int
    min_turnover: float
    min_price: float
    lookback_years: int


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    window_range: List[int]
    k_low_range: List[float]
    max_hold: int
    min_hit_rate: float


@dataclass(frozen=True)
class WalkForwardConfig:
    is_years: int
    oos_years: int
    holdout_years: int


@dataclass(frozen=True)
class SlippageConfig:
    gap_2pct: float
    gap_5pct: float
    gap_high: float


@dataclass(frozen=True)
class ExecutionConfig:
    circuit_guard_pct: float
    fees_bps: float
    slippage_model: SlippageConfig


@dataclass(frozen=True)
class PortfolioConfig:
    max_concurrent: int
    position_size: float
    equal_weight: bool
    reentry_lockout: bool


@dataclass(frozen=True)
class ReportingConfig:
    generate_plots: bool
    output_formats: List[Literal["json", "markdown", "csv"]]
    include_unfilled: bool


# §2. Top-Level Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """The root configuration object, composing all nested sections."""
    run: RunConfig
    data: DataConfig
    universe: UniverseConfig
    detector: DetectorConfig
    walk_forward: WalkForwardConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    reporting: ReportingConfig


# §3. Validation and Loading
# --------------------------------------------------------------------------------------


def _from_dict(data_class: Type[Any], data: Any) -> Any:
    """Recursively creates nested dataclasses from a dictionary."""
    if isinstance(data, dict):
        # We can't know the type of data_class at static analysis time, so we
        # have to treat it as Any. The caller is responsible for casting.
        field_types = {f.name: f.type for f in data_class.__dataclass_fields__.values()}

        kwargs = {}
        for k, v in data.items():
            field_type = field_types.get(k)
            # If the key from the dict is a field in the dataclass, recurse.
            # Otherwise, just pass it through. The dataclass constructor will
            # raise a TypeError for unexpected arguments, which is handled
            # by the caller. This satisfies mypy's check for None as an arg.
            kwargs[k] = _from_dict(field_type, v) if field_type else v
        return data_class(**kwargs)

    # Convert date strings to date objects
    if isinstance(data, str) and data_class is date:
        return date.fromisoformat(data)
    # Convert path strings to Path objects
    if isinstance(data, str) and data_class is Path:
        return Path(data)
    return data


def _validate_config(cfg: Dict[str, Any]) -> None:
    """
    Performs simple, explicit validation checks on the raw config dictionary.
    Fail fast on any logical inconsistencies.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML object.")

    # Date validation
    run_t0 = date.fromisoformat(cfg["run"]["t0"])
    data_start = date.fromisoformat(cfg["data"]["start_date"])
    data_end = date.fromisoformat(cfg["data"]["end_date"])

    if data_end <= data_start:
        raise ValueError("data.end_date must be after data.start_date")

    if not (data_start < run_t0 < data_end):
        raise ValueError("run.t0 must be within the data start and end dates")

    # Walk-forward validation
    wf_config = cfg["walk_forward"]
    total_wf_years = wf_config["is_years"] + wf_config["oos_years"]
    if total_wf_years <= 0:
        raise ValueError("Walk-forward years (is_years + oos_years) must be positive.")

    # Add any other critical checks here.


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
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {config_path}: {e}") from e

    # Perform validation before trying to create the objects
    _validate_config(raw_config)

    # Convert the raw dictionary to nested dataclasses
    try:
        # We cast here because _from_dict is too dynamic for mypy to track types.
        # The validation above gives us confidence that the structure is correct.
        return cast(Config, _from_dict(Config, raw_config))
    except (TypeError, KeyError) as e:
        raise ValueError(f"Configuration validation failed: missing or invalid key. Details: {e}") from e
