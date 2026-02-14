"""Anomaly scoring and map extraction."""

from __future__ import annotations

import torch
from torch.nn import functional


def anomaly_heatmap(x: torch.Tensor, x_rebuilt: torch.Tensor) -> torch.Tensor:
    """Absolute reconstruction residual heatmap, normalized per sample."""
    diff = torch.abs(x - x_rebuilt).mean(dim=1, keepdim=True)
    flat = diff.flatten(start_dim=1)
    min_val = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_val = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (diff - min_val) / (max_val - min_val + 1e-8)


def anomaly_score_l1(x: torch.Tensor, x_rebuilt: torch.Tensor) -> torch.Tensor:
    """Per-sample scalar score based on mean absolute residual."""
    return torch.abs(x - x_rebuilt).mean(dim=(1, 2, 3))


def anomaly_score_ssim_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-sample SSIM distance (1 - SSIM)."""
    c1 = 0.01**2
    c2 = 0.03**2
    window_size = 11
    pad = window_size // 2

    mu_x = functional.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = functional.avg_pool2d(y, window_size, stride=1, padding=pad)

    sigma_x = (
        functional.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x * mu_x
    )
    sigma_y = (
        functional.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y * mu_y
    )
    sigma_xy = (
        functional.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_x * mu_y
    )

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)

    ssim_per_sample = ssim_map.mean(dim=(1, 2, 3))
    return 1.0 - ssim_per_sample
