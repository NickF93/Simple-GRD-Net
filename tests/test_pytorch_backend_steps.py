from pathlib import Path

import pytest
import torch

from grdnet.backends.registry import create_backend
from grdnet.config.loader import load_experiment_config


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


def _batch() -> dict[str, torch.Tensor]:
    image = torch.rand((1, 1, 32, 32), dtype=torch.float32)
    roi_mask = torch.ones_like(image)
    gt_mask = torch.zeros_like(image)
    return {
        "image": image,
        "roi_mask": roi_mask,
        "gt_mask": gt_mask,
    }


def test_train_step_and_eval_step_smoke() -> None:
    backend = create_backend(_lightweight_train_cfg())

    train_output = backend.train_step(_batch())
    assert "loss.generator" in train_output.stats
    assert train_output.patch_scores.ndim == 1
    assert train_output.heatmap.ndim == 4

    eval_output = backend.eval_step(_batch())
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
