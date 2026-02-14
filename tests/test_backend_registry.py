from types import SimpleNamespace

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


def test_registry_loads_only_pytorch_when_requested(monkeypatch) -> None:
    calls: list[str] = []

    class _FakePyTorchBackend:
        def __init__(self, cfg) -> None:
            self.cfg = cfg

    def _fake_import(name: str):
        calls.append(name)
        if name == "grdnet.backends.pytorch_backend":
            return SimpleNamespace(PyTorchBackend=_FakePyTorchBackend)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr("grdnet.backends.registry.import_module", _fake_import)

    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.backend.name = "pytorch"
    backend = create_backend(cfg)

    assert isinstance(backend, _FakePyTorchBackend)
    assert calls == ["grdnet.backends.pytorch_backend"]


def test_registry_loads_only_tf_scaffold_when_requested(monkeypatch) -> None:
    calls: list[str] = []

    class _FakeTensorFlowBackend:
        def __init__(self, cfg) -> None:
            self.cfg = cfg

    def _fake_import(name: str):
        calls.append(name)
        if name == "grdnet.backends.tensorflow_backend":
            return SimpleNamespace(TensorFlowScaffoldBackend=_FakeTensorFlowBackend)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr("grdnet.backends.registry.import_module", _fake_import)

    cfg = load_experiment_config("configs/profiles/deepindustrial_sn_2026.yaml")
    cfg.backend.name = "tensorflow_scaffold"
    backend = create_backend(cfg)

    assert isinstance(backend, _FakeTensorFlowBackend)
    assert calls == ["grdnet.backends.tensorflow_backend"]
