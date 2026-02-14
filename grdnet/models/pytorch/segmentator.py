"""U-Net segmentator used in full GRD-Net profile."""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two convolution-BN-ReLU layers."""
        return self.block(x)


class UNetSegmentator(nn.Module):
    """Compact U-Net for anomaly map prediction."""

    def __init__(self, in_channels: int, base_features: int) -> None:
        super().__init__()
        self.enc1 = _ConvBlock(in_channels, base_features)
        self.enc2 = _ConvBlock(base_features, base_features * 2)
        self.enc3 = _ConvBlock(base_features * 2, base_features * 4)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mid = _ConvBlock(base_features * 4, base_features * 8)

        self.up3 = nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = _ConvBlock(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = _ConvBlock(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(
            base_features * 2,
            base_features,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = _ConvBlock(base_features * 2, base_features)

        self.out = nn.Sequential(
            nn.Conv2d(base_features, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a one-channel anomaly mask from concatenated inputs."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        mid = self.mid(self.pool(e3))

        d3 = self.up3(mid)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)
