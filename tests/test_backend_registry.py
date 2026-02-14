import pytest

from grdnet.backends.registry import create_backend
from grdnet.config.loader import load_experiment_config


def test_tf_scaffold_raises() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.backend.name = "tensorflow_scaffold"
    cfg.backend.device = "cpu"
    backend = create_backend(cfg)

    with pytest.raises(NotImplementedError):
        backend.build_models()
