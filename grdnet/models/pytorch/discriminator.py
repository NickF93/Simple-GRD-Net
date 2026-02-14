"""Discriminator model for adversarial feature matching."""

from __future__ import annotations

import torch
from torch import nn

from grdnet.models.pytorch.blocks import ResidualEncoder


class Discriminator(nn.Module):
    """Convolutional discriminator returning feature map + real/fake logits."""

    def __init__(
        self,
        *,
        in_channels: int,
        base_features: int,
        stages: tuple[int, ...],
        image_size: int,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(in_channels, base_features, stages)

        down_factor = 2 ** len(stages)
        spatial = image_size // down_factor
        dense_in = self.encoder.out_channels * spatial * spatial

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_in, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits
