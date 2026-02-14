from pathlib import Path

import pytest

from grdnet.config.loader import load_experiment_config
from grdnet.core.exceptions import ConfigurationError


def test_profile_config_loads() -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    assert cfg.profile.mode == "deepindustrial_sn_2026"
    assert cfg.profile.use_segmentator is False
    assert len(cfg.model.stages) == 4
    assert all(stage >= 1 for stage in cfg.model.stages)
    assert cfg.model.base_features >= 8
    assert cfg.losses.contextual_base == "huber"
    assert cfg.losses.use_noise_regularization is True


def test_missing_config_file_raises() -> None:
    with pytest.raises(ConfigurationError, match="Config file not found"):
        _ = load_experiment_config(Path("configs/profiles/does_not_exist.yaml"))


def test_invalid_config_raises(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("profile: {}", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid config"):
        _ = load_experiment_config(invalid)
