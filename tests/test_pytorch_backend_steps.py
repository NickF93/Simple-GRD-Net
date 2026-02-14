import warnings
from pathlib import Path

import pytest
import torch

from grdnet.backends.registry import create_backend
from grdnet.config.loader import load_experiment_config
from grdnet.core.exceptions import ConfigurationError


def _lightweight_train_cfg():
    cfg = load_experiment_config(Path("configs/profiles/grdnet_2023_full.yaml"))
    cfg.backend.name = "pytorch"
    cfg.backend.device = "cpu"
    cfg.data.channels = 1
    cfg.data.image_size = 32
    cfg.data.patch_size = (32, 32)
    cfg.data.patch_stride = (32, 32)
    cfg.model.base_features = 8
    cfg.model.stages = (1, 1)
    cfg.model.latent_dim = 8
    cfg.model.segmentator_base_features = 8
    cfg.model.dense_bottleneck = False
    cfg.augmentation.perlin_probability = 0.0
    cfg.augmentation.gaussian_noise_std_max = 0.0
    return cfg


def _lightweight_runtime_cfg():
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.backend.name = "pytorch"
    cfg.backend.device = "cpu"
    cfg.data.image_size = 32
    cfg.data.patch_size = (32, 32)
    cfg.data.patch_stride = (32, 32)
    cfg.model.base_features = 8
    cfg.model.stages = (1, 1)
    cfg.model.latent_dim = 8
    cfg.model.dense_bottleneck = False
    cfg.augmentation.perlin_probability = 0.0
    cfg.augmentation.gaussian_noise_std_max = 0.0
    return cfg


def _batch(*, channels: int) -> dict[str, torch.Tensor]:
    image = torch.rand((1, channels, 32, 32), dtype=torch.float32)
    roi_mask = torch.ones_like(image)
    gt_mask = torch.zeros_like(image)
    return {
        "image": image,
        "roi_mask": roi_mask,
        "gt_mask": gt_mask,
    }


def _disable_backward_step(backend) -> None:
    def _no_backward_step(**_kwargs) -> None:
        return None

    backend._backward_step = _no_backward_step


def test_train_step_and_eval_step_smoke() -> None:
    backend = create_backend(_lightweight_train_cfg())
    _disable_backward_step(backend)

    train_output = backend.train_step(_batch(channels=backend.cfg.data.channels))
    assert "loss.generator" in train_output.stats
    assert train_output.patch_scores.ndim == 1
    assert train_output.heatmap.ndim == 4

    eval_output = backend.eval_step(_batch(channels=backend.cfg.data.channels))
    assert "loss.total_eval" in eval_output.stats
    assert eval_output.patch_scores.ndim == 1
    assert eval_output.heatmap.ndim == 4


def test_prepare_requires_tensor_fields() -> None:
    backend = create_backend(_lightweight_train_cfg())

    with pytest.raises(TypeError, match="batch\\['image'\\]"):
        backend._prepare(
            {"image": 1, "roi_mask": torch.ones(1), "gt_mask": torch.ones(1)}
        )

    with pytest.raises(TypeError, match="batch\\['roi_mask'\\]"):
        backend._prepare(
            {
                "image": torch.rand((1, 1, 32, 32)),
                "roi_mask": 1,
                "gt_mask": torch.ones((1, 1, 32, 32)),
            }
        )

    with pytest.raises(TypeError, match="batch\\['gt_mask'\\]"):
        backend._prepare(
            {
                "image": torch.rand((1, 1, 32, 32)),
                "roi_mask": torch.ones((1, 1, 32, 32)),
                "gt_mask": 1,
            }
        )


def test_train_and_eval_step_without_segmentator_branch() -> None:
    backend = create_backend(_lightweight_runtime_cfg())
    _disable_backward_step(backend)

    train_output = backend.train_step(_batch(channels=backend.cfg.data.channels))
    assert train_output.seg_map is None
    assert train_output.stats["loss.segmentator"] == 0.0

    eval_output = backend.eval_step(_batch(channels=backend.cfg.data.channels))
    assert eval_output.seg_map is None
    assert eval_output.stats["loss.segmentator"] == 0.0


def test_resolve_device_auto_branch(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert str(create_backend(_lightweight_train_cfg()).device) == "cpu"


def test_resolve_device_auto_falls_back_on_cuda_warning(monkeypatch) -> None:
    monkeypatch.setattr("torch.backends.cuda.is_built", lambda: True)

    def _warn_then_false() -> bool:
        warnings.warn("cuda probe failed", UserWarning, stacklevel=1)
        return False

    monkeypatch.setattr("torch.cuda.is_available", _warn_then_false)
    assert str(create_backend(_lightweight_train_cfg()).device) == "cpu"


def test_resolve_device_cuda_raises_on_probe_warning(monkeypatch) -> None:
    cfg = _lightweight_train_cfg()
    cfg.backend.device = "cuda"
    monkeypatch.setattr("torch.backends.cuda.is_built", lambda: True)

    def _warn_then_false() -> bool:
        warnings.warn("cuda probe failed", UserWarning, stacklevel=1)
        return False

    monkeypatch.setattr("torch.cuda.is_available", _warn_then_false)
    with pytest.raises(ConfigurationError, match="initialization failed"):
        _ = create_backend(cfg)


def test_autocast_mode_switches_with_amp_flag(monkeypatch) -> None:
    backend = create_backend(_lightweight_train_cfg())
    assert backend._autocast().__class__.__name__ == "nullcontext"

    class _FakeAutocast:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            _ = exc_type, exc, tb

    monkeypatch.setattr("torch.autocast", lambda **kwargs: _FakeAutocast())
    backend._amp_enabled = True
    assert backend._autocast().__class__.__name__ == "_FakeAutocast"


class _FakeLoss:
    def __init__(self) -> None:
        self.backward_called = False

    def backward(self) -> None:
        self.backward_called = True


class _FakeScaled:
    def __init__(self, loss: _FakeLoss) -> None:
        self._loss = loss

    def backward(self) -> None:
        self._loss.backward()


class _FakeScaler:
    def __init__(self) -> None:
        self.unscaled = False
        self.stepped = False
        self.updated = False

    def scale(self, loss: _FakeLoss) -> _FakeScaled:
        return _FakeScaled(loss)

    def unscale_(self, optimizer) -> None:
        _ = optimizer
        self.unscaled = True

    def step(self, optimizer) -> None:
        _ = optimizer
        self.stepped = True

    def update(self) -> None:
        self.updated = True


def test_backward_step_non_amp_and_amp_paths(monkeypatch) -> None:
    backend = create_backend(_lightweight_train_cfg())
    parameter = next(backend.models.generator.parameters())
    optimizer = backend.optimizers.generator

    called = {"step": 0}

    def _count_step() -> None:
        called["step"] += 1

    monkeypatch.setattr(optimizer, "step", _count_step)
    fake_loss = _FakeLoss()
    backend._amp_enabled = False
    backend._backward_step(
        loss=fake_loss,  # type: ignore[arg-type]
        optimizer=optimizer,
        parameters=[parameter],
        max_grad_norm=1.0,
    )
    assert fake_loss.backward_called is True
    assert called["step"] == 1

    backend._amp_enabled = True
    backend._grad_scaler = None
    with pytest.raises(RuntimeError, match="GradScaler"):
        backend._backward_step(
            loss=fake_loss,  # type: ignore[arg-type]
            optimizer=optimizer,
        )

    fake_scaler = _FakeScaler()
    backend._grad_scaler = fake_scaler  # type: ignore[assignment]
    backend._backward_step(
        loss=fake_loss,  # type: ignore[arg-type]
        optimizer=optimizer,
        parameters=[parameter],
        max_grad_norm=1.0,
    )
    assert fake_scaler.unscaled is True
    assert fake_scaler.stepped is True
    assert fake_scaler.updated is True
