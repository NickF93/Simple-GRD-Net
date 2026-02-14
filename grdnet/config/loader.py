"""YAML config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from grdnet.config.schema import ExperimentConfig
from grdnet.core.exceptions import ConfigurationError


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate experiment config from YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    try:
        config = ExperimentConfig.model_validate(payload)
    except Exception as exc:  # pydantic provides detailed message.
        raise ConfigurationError(f"Invalid config at {config_path}: {exc}") from exc

    return config
