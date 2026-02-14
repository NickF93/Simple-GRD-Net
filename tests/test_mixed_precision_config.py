import pytest

from grdnet.backends.registry import create_backend
from grdnet.config.loader import load_experiment_config
from grdnet.core.exceptions import ConfigurationError


def test_mixed_precision_requires_cuda_device() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.backend.name = "pytorch"
    cfg.backend.device = "cpu"
    cfg.backend.mixed_precision = True

    with pytest.raises(ConfigurationError):
        _ = create_backend(cfg)
