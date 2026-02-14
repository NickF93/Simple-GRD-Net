"""Discriminator model for adversarial feature matching."""

from __future__ import annotations

from typing import Literal

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
        image_shape: tuple[int, int],
        encoder_downsample_position: Literal["first", "last"] = "last",
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(
            in_channels,
            base_features,
            stages,
            downsample_position=encoder_downsample_position,
        )

        down_factor = 2 ** len(stages)
        spatial_h = image_shape[0] // down_factor
        spatial_w = image_shape[1] // down_factor
        dense_in = self.encoder.out_channels * spatial_h * spatial_w

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_in, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return feature embedding and real/fake logits."""
        features = self.encoder(x)
        logits = self.classifier(features)
        return features, logits
