from pathlib import Path

from grdnet.config.loader import load_experiment_config


def test_profile_config_loads() -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    assert cfg.profile.mode == "deepindustrial_sn_2026"
    assert cfg.profile.use_segmentator is False
