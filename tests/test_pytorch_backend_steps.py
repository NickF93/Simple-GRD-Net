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
    cfg.backend.mixed_precision = False
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


def test_train_step_emits_preview_payload_when_enabled() -> None:
    cfg = _lightweight_runtime_cfg()
    cfg.reporting.train_batch_preview.enabled = True
    backend = create_backend(cfg)
    _disable_backward_step(backend)

    output = backend.train_step(_batch(channels=backend.cfg.data.channels))
    assert output.train_batch_preview is not None
    preview = output.train_batch_preview
    assert preview.x.shape == preview.x_noisy.shape
    assert preview.x.shape == preview.x_rebuilt.shape
    assert preview.noise_mask.shape[0] == preview.x.shape[0]
    assert preview.noise_mask.shape[2:] == preview.x.shape[2:]


def test_train_step_uses_two_generator_passes_and_phase_train_modes(
    monkeypatch,
) -> None:
    backend = create_backend(_lightweight_train_cfg())
    _disable_backward_step(backend)

    generator_calls: list[tuple[bool, bool]] = []
    discriminator_calls: list[tuple[bool, bool, bool]] = []
    segmentator_calls: list[bool] = []

    orig_generator_forward = backend.models.generator.forward
    orig_discriminator_forward = backend.models.discriminator.forward
    if backend.models.segmentator is None:
        raise AssertionError("Expected segmentator branch in GRD config")
    orig_segmentator_forward = backend.models.segmentator.forward

    def _generator_forward_spy(x: torch.Tensor):
        generator_calls.append(
            (
                backend.models.generator.training,
                torch.is_grad_enabled(),
            )
        )
        return orig_generator_forward(x)

    def _discriminator_forward_spy(x: torch.Tensor):
        discriminator_calls.append(
            (
                backend.models.discriminator.training,
                torch.is_grad_enabled(),
                any(
                    parameter.requires_grad
                    for parameter in backend.models.discriminator.parameters()
                ),
            )
        )
        return orig_discriminator_forward(x)

    def _segmentator_forward_spy(x: torch.Tensor):
        segmentator_calls.append(backend.models.segmentator.training)
        return orig_segmentator_forward(x)

    monkeypatch.setattr(backend.models.generator, "forward", _generator_forward_spy)
    monkeypatch.setattr(
        backend.models.discriminator,
        "forward",
        _discriminator_forward_spy,
    )
    monkeypatch.setattr(
        backend.models.segmentator,
        "forward",
        _segmentator_forward_spy,
    )

    _ = backend.train_step(_batch(channels=backend.cfg.data.channels))

    assert len(generator_calls) == 2
    assert generator_calls[0] == (False, False)
    assert generator_calls[1] == (True, True)

    assert len(discriminator_calls) == 4
    assert discriminator_calls[0] == (True, True, True)
    assert discriminator_calls[1] == (True, True, True)
    assert discriminator_calls[2] == (False, False, False)
    assert discriminator_calls[3] == (False, True, False)

    assert segmentator_calls == [True]
    discriminator_params = backend.models.discriminator.parameters()
    assert all(parameter.requires_grad for parameter in discriminator_params)


def test_resolve_device_auto_branch(monkeypatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    cfg = _lightweight_train_cfg()
    cfg.backend.device = "auto"
    assert str(create_backend(cfg).device) == "cpu"


def test_resolve_device_auto_raises_on_cuda_probe_warning(monkeypatch) -> None:
    monkeypatch.setattr("torch.backends.cuda.is_built", lambda: True)

    def _warn_then_false() -> bool:
        raise UserWarning("cuda probe failed")

    monkeypatch.setattr("torch.cuda.is_available", _warn_then_false)
    cfg = _lightweight_train_cfg()
    cfg.backend.device = "auto"
    with pytest.raises(ConfigurationError, match="auto'.*initialization failed"):
        _ = create_backend(cfg)


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
