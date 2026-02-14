from pathlib import Path

import torch
from torch import nn

from grdnet.backends.registry import create_backend
from grdnet.config.loader import load_experiment_config
from grdnet.metrics.anomaly import anomaly_score_ssim_per_sample


class _GeneratorStub(nn.Module):
    def __init__(self, rebuilt: torch.Tensor) -> None:
        super().__init__()
        self._rebuilt = rebuilt

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.zeros((x.shape[0], 4), dtype=x.dtype, device=x.device)
        rebuilt = self._rebuilt.to(device=x.device, dtype=x.dtype)
        return z, rebuilt, z


class _SegmentatorStub(nn.Module):
    def __init__(self, seg_map: torch.Tensor) -> None:
        super().__init__()
        self._seg_map = seg_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = x
        return self._seg_map


def _use_lightweight_model(cfg) -> None:
    cfg.backend.mixed_precision = False
    cfg.model.base_features = 8
    cfg.model.stages = (1, 1)
    cfg.model.latent_dim = 8
    cfg.model.segmentator_base_features = 8
    cfg.model.dense_bottleneck = False
    cfg.data.image_size = 16
    cfg.data.patch_size = (16, 16)
    cfg.data.patch_stride = (16, 16)


def test_grd_profile_default_uses_segmentator_roi_max(monkeypatch) -> None:
    cfg = load_experiment_config(Path("configs/profiles/grdnet_2023_full.yaml"))
    cfg.backend.name = "pytorch"
    cfg.backend.device = "cpu"
    cfg.data.channels = 1
    cfg.inference.scoring_strategy = "profile_default"
    _use_lightweight_model(cfg)

    backend = create_backend(cfg)
    x = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    roi_mask = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]]],
            [[[1.0, 1.0], [0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    x_rebuilt = torch.full_like(x, 0.5)
    seg_map = torch.tensor(
        [
            [[[0.2, 0.8], [0.5, 0.7]]],
            [[[0.1, 0.9], [0.6, 0.4]]],
        ],
        dtype=torch.float32,
    )
    expected_scores = (seg_map * roi_mask).flatten(start_dim=1).amax(dim=1)

    backend.models.generator = _GeneratorStub(x_rebuilt)
    backend.models.segmentator = _SegmentatorStub(seg_map)
    monkeypatch.setattr(
        backend,
        "_prepare",
        lambda batch: (x, roi_mask, torch.zeros_like(roi_mask)),
    )

    output = backend.infer_step(
        {"image": x, "roi_mask": roi_mask, "gt_mask": torch.zeros_like(roi_mask)}
    )
    assert torch.allclose(output.patch_scores, expected_scores)


def test_deepindustrial_profile_default_uses_ssim(monkeypatch) -> None:
    cfg = load_experiment_config(Path("configs/profiles/deepindustrial_sn_2026.yaml"))
    cfg.backend.name = "pytorch"
    cfg.backend.device = "cpu"
    cfg.inference.scoring_strategy = "profile_default"
    _use_lightweight_model(cfg)

    backend = create_backend(cfg)
    x = torch.tensor(
        [
            [[[0.0, 0.2], [0.3, 0.4]]],
            [[[0.1, 0.1], [0.1, 0.1]]],
        ],
        dtype=torch.float32,
    )
    roi_mask = torch.ones_like(x)
    x_rebuilt = x * 0.5
    expected_scores = anomaly_score_ssim_per_sample(x, x_rebuilt)

    backend.models.generator = _GeneratorStub(x_rebuilt)
    monkeypatch.setattr(
        backend,
        "_prepare",
        lambda batch: (x, roi_mask, torch.zeros_like(roi_mask)),
    )

    output = backend.infer_step(
        {"image": x, "roi_mask": roi_mask, "gt_mask": torch.zeros_like(roi_mask)}
    )
    assert torch.allclose(output.patch_scores, expected_scores)


def test_grd_profile_supports_explicit_ssim_override(monkeypatch) -> None:
    cfg = load_experiment_config(Path("configs/profiles/grdnet_2023_full.yaml"))
    cfg.backend.name = "pytorch"
    cfg.backend.device = "cpu"
    cfg.data.channels = 1
    cfg.inference.scoring_strategy = "ssim"
    _use_lightweight_model(cfg)

    backend = create_backend(cfg)
    x = torch.tensor(
        [[[[0.9, 0.9], [0.9, 0.9]]]],
        dtype=torch.float32,
    )
    roi_mask = torch.ones_like(x)
    x_rebuilt = x * 0.8
    expected_scores = anomaly_score_ssim_per_sample(x, x_rebuilt)

    backend.models.generator = _GeneratorStub(x_rebuilt)
    backend.models.segmentator = _SegmentatorStub(torch.zeros_like(x))
    monkeypatch.setattr(
        backend,
        "_prepare",
        lambda batch: (x, roi_mask, torch.zeros_like(roi_mask)),
    )

    output = backend.infer_step(
        {"image": x, "roi_mask": roi_mask, "gt_mask": torch.zeros_like(roi_mask)}
    )
    assert torch.allclose(output.patch_scores, expected_scores)
