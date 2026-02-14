"""Reusable PyTorch residual building blocks."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Basic residual block with optional projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one residual block."""
        residual = self.proj(inputs)
        x = self.act(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        x = self.act(x + residual)
        return x


class ResidualEncoder(nn.Module):
    """Configurable residual encoder used by generator and discriminator."""

    def __init__(
        self,
        in_channels: int,
        base_features: int,
        stages: tuple[int, ...],
        *,
        downsample_position: Literal["first", "last"] = "last",
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(base_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks: list[nn.Module] = []
        channels = base_features
        for stage_idx, n_blocks in enumerate(stages):
            stage_channels = base_features * (2**stage_idx)
            for block_idx in range(n_blocks):
                is_downsample = (
                    block_idx == 0
                    if downsample_position == "first"
                    else block_idx == n_blocks - 1
                )
                stride = 2 if is_downsample else 1
                blocks.append(ResidualBlock(channels, stage_channels, stride=stride))
                channels = stage_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an input tensor into stage-wise residual features."""
        x = self.stem(x)
        x = self.blocks(x)
        return x


class ResidualDecoder(nn.Module):
    """Residual decoder mirroring encoder stage structure."""

    def __init__(
        self,
        out_channels: int,
        base_features: int,
        stages: tuple[int, ...],
        bottleneck_channels: int,
        *,
        upsample_position: Literal["first", "last"] = "last",
    ) -> None:
        super().__init__()

        rev_stages = tuple(reversed(stages))
        channels = bottleneck_channels
        modules: list[nn.Module] = []

        for stage_idx, n_blocks in enumerate(rev_stages):
            target_channels = base_features * (2 ** (len(rev_stages) - stage_idx - 1))
            for block_idx in range(n_blocks):
                is_upsample = (
                    block_idx == 0
                    if upsample_position == "first"
                    else block_idx == n_blocks - 1
                )
                if is_upsample:
                    modules.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                channels,
                                target_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(target_channels),
                            nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
                else:
                    modules.append(ResidualBlock(channels, target_channels, stride=1))
                channels = target_channels

        self.blocks = nn.Sequential(*modules)
        self.out = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode residual features into a reconstructed image patch."""
        x = self.blocks(x)
        x = self.out(x)
        return x
