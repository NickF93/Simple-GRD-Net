import pytest
import torch

from grdnet.config.schema import AugmentationConfig
from grdnet.training.perturbation import (
    apply_gaussian_noise,
    apply_geometry_augmentation,
    apply_perlin_perturbation,
)


def _aug_cfg(**kwargs) -> AugmentationConfig:
    defaults = {
        "perlin_probability": 1.0,
        "perlin_min_area": 1,
        "rotation_degrees": (0.0, 0.0),
        "horizontal_flip_probability": 0.0,
        "vertical_flip_probability": 0.0,
        "gaussian_noise_std_max": 0.1,
    }
    defaults.update(kwargs)
    return AugmentationConfig(**defaults)


def test_geometry_augmentation_noop_when_disabled() -> None:
    cfg = _aug_cfg()
    images = torch.rand((2, 1, 8, 8), dtype=torch.float32)
    roi_masks = (torch.rand((2, 1, 8, 8)) > 0.5).float()

    out_images, out_masks = apply_geometry_augmentation(images, roi_masks, cfg)
    assert torch.allclose(out_images, images)
    assert torch.allclose(out_masks, roi_masks)


def test_perlin_perturbation_returns_expected_shapes_and_ranges() -> None:
    cfg = _aug_cfg(perlin_probability=1.0, perlin_min_area=1)
    images = torch.zeros((2, 1, 16, 16), dtype=torch.float32)

    x_noisy, noise, mask, beta = apply_perlin_perturbation(images, cfg)
    assert x_noisy.shape == images.shape
    assert noise.shape == images.shape
    assert mask.shape == (2, 1, 16, 16)
    assert beta.shape == (2, 1, 1, 1)
    assert float(x_noisy.min().item()) >= 0.0
    assert float(x_noisy.max().item()) <= 1.0
    assert float(beta.min().item()) >= 0.5
    assert float(beta.max().item()) <= 1.0


def test_gaussian_noise_noop_when_sigma_zero() -> None:
    images = torch.rand((2, 1, 8, 8), dtype=torch.float32)
    out = apply_gaussian_noise(images, sigma_max=0.0)
    assert torch.allclose(out, images)


def test_perlin_perturbation_rejects_min_area_above_patch_area() -> None:
    cfg = _aug_cfg(perlin_min_area=65)
    images = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

    with pytest.raises(ValueError, match="perlin_min_area exceeds patch area"):
        _ = apply_perlin_perturbation(images, cfg)
