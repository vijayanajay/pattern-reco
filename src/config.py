"""
Configuration loading and validation.

Simplifies the original design by using a single Pydantic model
and leveraging dictionaries for nested configuration, reducing boilerplate.
"""

from datetime import date
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, ValidationError, root_validator

__all__ = ["load_config", "Config"]


class Config(BaseModel):
    """
    A single Pydantic model for the entire configuration.
    Nested structures are handled as dictionaries, providing flexibility.
    """

    run: Dict[str, Any]
    data: Dict[str, Any]
    universe: Dict[str, Any]
    detector: Dict[str, Any]
    walk_forward: Dict[str, Any]
    execution: Dict[str, Any]
    portfolio: Dict[str, Any]
    reporting: Dict[str, Any]

    class Config:
        # Allow Pydantic to automatically convert date strings.
        json_encoders = {date: lambda v: v.strftime("%Y-%m-%d")}

    @root_validator(pre=True)
    def check_required_sections(cls, values: Dict) -> Dict:
        """Check for presence of all top-level configuration sections."""
        required = {
            "run",
            "data",
            "universe",
            "detector",
            "walk_forward",
            "execution",
            "portfolio",
            "reporting",
        }
        missing = required - set(values.keys())
        if missing:
            raise ValueError(f"Missing required config sections: {', '.join(missing)}")
        return values

    @root_validator
    def validate_and_parse_dates(cls, values: Dict) -> Dict:
        """
        Validate date logic and parse date strings, since they are in a generic Dict.
        Pydantic v1 doesn't automatically parse types inside a `Dict[str, Any]`.
        """
        from datetime import datetime

        run_cfg = values.get("run", {})
        data_cfg = values.get("data", {})

        try:
            # Manually parse dates from strings
            run_t0_str = run_cfg.get("t0")
            data_start_str = data_cfg.get("start_date")
            data_end_str = data_cfg.get("end_date")

            if not all([run_t0_str, data_start_str, data_end_str]):
                raise ValueError("Missing required date fields: run.t0, data.start_date, data.end_date")

            run_t0 = datetime.strptime(str(run_t0_str), "%Y-%m-%d").date()
            data_start = datetime.strptime(str(data_start_str), "%Y-%m-%d").date()
            data_end = datetime.strptime(str(data_end_str), "%Y-%m-%d").date()

            # Store the parsed `date` objects back into the config dict
            values["run"]["t0"] = run_t0
            values["data"]["start_date"] = data_start
            values["data"]["end_date"] = data_end

        except (ValueError, TypeError) as e:
            # Catches missing keys from .get() and strptime format errors
            raise ValueError(f"Invalid or missing date configuration: {e}") from e

        if data_end <= data_start:
            raise ValueError("data.end_date must be after data.start_date")
        if not (data_start < run_t0 < data_end):
            raise ValueError("run.t0 must be within the data.start_date and data.end_date")

        return values


# impure
def load_config(config_path: Path) -> Config:
    """
    Load and validate YAML configuration file.
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
        # Re-raise Pydantic's error to provide more specific feedback.
        raise ValueError(f"Configuration validation failed: {e}") from e
