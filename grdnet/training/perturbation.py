"""Training-time perturbation and augmentation utilities."""

from __future__ import annotations

import math

import torch
from torchvision.transforms.functional import InterpolationMode, rotate

from grdnet.config.schema import AugmentationConfig


def _lerp(a: torch.Tensor, b: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return a + weight * (b - a)


def _perlin_2d(
    height: int,
    width: int,
    res_h: int,
    res_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a 2D Perlin map in [-1, 1]."""
    delta_h = res_h / height
    delta_w = res_w / width

    grid = torch.stack(
        torch.meshgrid(
            torch.arange(0, res_h, delta_h, device=device),
            torch.arange(0, res_w, delta_w, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    grid = grid[:height, :width] % 1.0

    angles = 2 * math.pi * torch.rand((res_h + 1, res_w + 1), device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    d_h = max(1, height // res_h)
    d_w = max(1, width // res_w)

    def tile(y0: int, y1: int, x0: int, x1: int) -> torch.Tensor:
        """Tile one gradient corner grid to full image resolution."""
        patch = gradients[y0:y1, x0:x1]
        return patch.repeat_interleave(d_h, dim=0).repeat_interleave(d_w, dim=1)

    def dot(grad: torch.Tensor, shift_y: float, shift_x: float) -> torch.Tensor:
        """Compute gradient dot product for one corner shift."""
        shifted = torch.stack((grid[..., 0] + shift_y, grid[..., 1] + shift_x), dim=-1)
        return (shifted * grad[:height, :width]).sum(dim=-1)

    n00 = dot(tile(0, -1, 0, -1), 0.0, 0.0)
    n10 = dot(tile(1, None, 0, -1), -1.0, 0.0)
    n01 = dot(tile(0, -1, 1, None), 0.0, -1.0)
    n11 = dot(tile(1, None, 1, None), -1.0, -1.0)

    def fade_curve(values: torch.Tensor) -> torch.Tensor:
        """Apply quintic Perlin interpolation curve."""
        return 6 * values**5 - 15 * values**4 + 10 * values**3

    t = fade_curve(grid[:height, :width])

    return math.sqrt(2.0) * _lerp(
        _lerp(n00, n10, t[..., 0]),
        _lerp(n01, n11, t[..., 0]),
        t[..., 1],
    )


def _sample_masked_noise(
    *,
    channels: int,
    height: int,
    width: int,
    min_area: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one Perlin perturbation and binary mask."""
    attempts = 8
    for _ in range(attempts):
        res_h = 2 ** int(torch.randint(0, 6, (1,), device=device).item())
        res_w = 2 ** int(torch.randint(0, 6, (1,), device=device).item())
        base = _perlin_2d(height, width, max(res_h, 1), max(res_w, 1), device=device)
        mask = (base > 0.5).to(dtype=torch.float32).unsqueeze(0)
        if int(mask.sum().item()) >= min_area:
            noise = base.unsqueeze(0).repeat(channels, 1, 1)
            noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
            return noise, mask

    raise RuntimeError(
        "Unable to generate Perlin mask satisfying min_area after 8 attempts."
    )


def apply_geometry_augmentation(
    images: torch.Tensor,
    roi_masks: torch.Tensor,
    cfg: AugmentationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply deterministic-shape augmentations sample-wise."""
    out_images = []
    out_masks = []
    for image_sample, roi_sample in zip(images, roi_masks, strict=True):
        # rotation range is profile-driven and defaults to the simple variant values.
        angle = float(torch.empty(1).uniform_(*cfg.rotation_degrees).item())
        augmented_image = rotate(
            image_sample,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
        )
        augmented_roi = rotate(
            roi_sample,
            angle=angle,
            interpolation=InterpolationMode.NEAREST,
        )

        if torch.rand(1).item() < cfg.horizontal_flip_probability:
            augmented_image = torch.flip(augmented_image, dims=[2])
            augmented_roi = torch.flip(augmented_roi, dims=[2])

        if torch.rand(1).item() < cfg.vertical_flip_probability:
            augmented_image = torch.flip(augmented_image, dims=[1])
            augmented_roi = torch.flip(augmented_roi, dims=[1])

        out_images.append(augmented_image)
        out_masks.append((augmented_roi > 0.5).to(dtype=torch.float32))

    return torch.stack(out_images, dim=0), torch.stack(out_masks, dim=0)


def apply_perlin_perturbation(
    images: torch.Tensor,
    cfg: AugmentationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Perlin perturbation with probability `q` and return full tuple.

    Returns
    -------
    x_noisy : torch.Tensor
    noise : torch.Tensor
    mask : torch.Tensor
    beta : torch.Tensor
    """
    bsz, channels, height, width = images.shape
    device = images.device
    if cfg.perlin_min_area > height * width:
        raise ValueError(
            "augmentation.perlin_min_area exceeds patch area; "
            "reduce perlin_min_area or increase patch size."
        )

    noise_batch = torch.zeros_like(images)
    mask_batch = torch.zeros((bsz, 1, height, width), device=device)

    for idx in range(bsz):
        if torch.rand(1, device=device).item() > cfg.perlin_probability:
            continue

        noise, mask = _sample_masked_noise(
            channels=channels,
            height=height,
            width=width,
            min_area=cfg.perlin_min_area,
            device=device,
        )
        noise_batch[idx] = noise
        mask_batch[idx] = mask

    beta = torch.empty((bsz, 1, 1, 1), device=device).uniform_(0.5, 1.0)
    x_inside = images * mask_batch
    x_outside = images * (1.0 - mask_batch)
    x_noisy = x_outside + ((1.0 - beta) * x_inside) + (beta * noise_batch)
    x_noisy = x_noisy.clamp(0.0, 1.0)

    return x_noisy, noise_batch, mask_batch, beta


def apply_gaussian_noise(images: torch.Tensor, sigma_max: float) -> torch.Tensor:
    """Apply sample-wise Gaussian noise and clamp back to [0, 1]."""
    if sigma_max <= 0.0:
        return images

    sigma = torch.empty(
        (images.shape[0], 1, 1, 1),
        device=images.device,
    ).uniform_(0.0, sigma_max)
    noise = torch.randn_like(images) * sigma
    return (images + noise).clamp(0.0, 1.0)
