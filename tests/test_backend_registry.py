import pytest

from grdnet.backends.registry import create_backend
from grdnet.config.loader import load_experiment_config
from grdnet.core.exceptions import BackendNotAvailableError


def test_tf_scaffold_raises() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.backend.name = "tensorflow_scaffold"
    cfg.backend.device = "cpu"
    backend = create_backend(cfg)

    with pytest.raises(NotImplementedError):
        backend.build_models()


def test_unknown_backend_name_raises() -> None:
    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.backend.name = "unknown_backend"  # type: ignore[assignment]

    with pytest.raises(BackendNotAvailableError):
        _ = create_backend(cfg)
