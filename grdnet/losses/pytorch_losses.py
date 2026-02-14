"""PyTorch loss implementations for both official profiles."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from grdnet.config.schema import ExperimentConfig


class SsimLoss(nn.Module):
    """Differentiable SSIM loss: 1 - mean(SSIM)."""

    def __init__(self, window_size: int = 11) -> None:
        super().__init__()
        self.window_size = window_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute SSIM distance `1 - mean(SSIM)`."""
        c1 = 0.01**2
        c2 = 0.03**2

        pad = self.window_size // 2
        mu_x = functional.avg_pool2d(x, self.window_size, stride=1, padding=pad)
        mu_y = functional.avg_pool2d(y, self.window_size, stride=1, padding=pad)

        sigma_x = (
            functional.avg_pool2d(
                x * x,
                self.window_size,
                stride=1,
                padding=pad,
            )
            - mu_x * mu_x
        )
        sigma_y = (
            functional.avg_pool2d(
                y * y,
                self.window_size,
                stride=1,
                padding=pad,
            )
            - mu_y * mu_y
        )
        sigma_xy = (
            functional.avg_pool2d(
                x * y,
                self.window_size,
                stride=1,
                padding=pad,
            )
            - mu_x * mu_y
        )

        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
        ssim_map = numerator / (denominator + 1e-8)
        return 1.0 - ssim_map.mean()


class FocalBinaryLoss(nn.Module):
    """Binary focal loss over probabilities in [0, 1]."""

    def __init__(self, alpha: float, gamma: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss for binary probabilities."""
        y_pred = y_pred.clamp(min=1e-6, max=1.0 - 1e-6)
        bce = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)
        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return focal.mean()


class GrdNetLossComputer:
    """Centralized loss orchestration for GRD-Net and Simple profiles."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.huber = nn.HuberLoss(reduction="mean", delta=1.0)
        self.l1 = nn.L1Loss(reduction="mean")
        self.l2 = nn.MSELoss(reduction="mean")
        self.ssim = SsimLoss(window_size=11)
        # AMP-safe discriminator objective (replaces Sigmoid + BCELoss).
        self.bce_logits = nn.BCEWithLogitsLoss(reduction="mean")
        self.focal = FocalBinaryLoss(
            alpha=cfg.losses.focal_alpha,
            gamma=cfg.losses.focal_gamma,
        )

    def contextual(self, x: torch.Tensor, x_rebuilt: torch.Tensor) -> torch.Tensor:
        """Contextual term with configurable profile-aligned base distance."""
        base_distance: torch.Tensor
        if self.cfg.losses.contextual_base == "huber":
            base_distance = self.huber(x_rebuilt, x)
        else:
            base_distance = self.l1(x_rebuilt, x)
        ssim = self.ssim(x_rebuilt, x)
        return self.cfg.losses.wa * base_distance + self.cfg.losses.wb * ssim

    def generator_total(
        self,
        *,
        x: torch.Tensor,
        x_rebuilt: torch.Tensor,
        z: torch.Tensor,
        z_rebuilt: torch.Tensor,
        feat_real: torch.Tensor,
        feat_fake: torch.Tensor,
        noise_loss: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted generator objective."""
        contextual = self.contextual(x, x_rebuilt)
        adversarial = self.l2(feat_fake, feat_real.detach())
        encoder = self.l1(z, z_rebuilt)

        total = (
            self.cfg.losses.w1 * adversarial
            + self.cfg.losses.w2 * contextual
            + self.cfg.losses.w3 * encoder
        )
        if self.cfg.losses.use_noise_regularization:
            total = total + (self.cfg.losses.w4 * noise_loss)
        stats = {
            "loss.contextual": float(contextual.detach().cpu().item()),
            "loss.adversarial": float(adversarial.detach().cpu().item()),
            "loss.encoder": float(encoder.detach().cpu().item()),
            "loss.noise": float(noise_loss.detach().cpu().item()),
            "loss.generator": float(total.detach().cpu().item()),
        }
        return total, stats

    def discriminator_total(
        self,
        pred_real_logits: torch.Tensor,
        pred_fake_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Binary discrimination loss over real/fake logits."""
        ones = torch.ones_like(pred_real_logits)
        zeros = torch.zeros_like(pred_fake_logits)
        loss = 0.5 * (
            self.bce_logits(pred_real_logits, ones)
            + self.bce_logits(pred_fake_logits, zeros)
        )
        return loss, float(loss.detach().cpu().item())

    def segmentator_total(
        self,
        *,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """ROI-intersection focal term from GRD-Net full profile."""
        pred_intersection = pred_mask * roi_mask
        loss = self.focal(pred_intersection, gt_mask)
        return loss, float(loss.detach().cpu().item())

    def anomaly_score(self, x: torch.Tensor, x_rebuilt: torch.Tensor) -> torch.Tensor:
        """Patch score used in DeepIndustrial-SN profile."""
        return self.ssim(x, x_rebuilt)
