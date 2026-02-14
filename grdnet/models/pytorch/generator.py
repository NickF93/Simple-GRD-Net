"""Generator model: encoder-decoder-encoder (GANomaly-style)."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from grdnet.models.pytorch.blocks import ResidualDecoder, ResidualEncoder


class GeneratorEDE(nn.Module):
    """Residual generator with explicit latent consistency head."""

    def __init__(
        self,
        *,
        in_channels: int,
        base_features: int,
        stages: tuple[int, ...],
        latent_dim: int,
        dense_bottleneck: bool,
        image_shape: tuple[int, int],
        encoder_downsample_position: Literal["first", "last"] = "last",
        decoder_upsample_position: Literal["first", "last"] = "last",
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(
            in_channels,
            base_features,
            stages,
            downsample_position=encoder_downsample_position,
        )
        self.encoder_reconstructed = ResidualEncoder(
            in_channels,
            base_features,
            stages,
            downsample_position=encoder_downsample_position,
        )

        self._dense_bottleneck = dense_bottleneck
        self._image_shape = image_shape
        self._latent_dim = latent_dim

        if dense_bottleneck:
            down_factor = 2 ** len(stages)
            spatial_h = image_shape[0] // down_factor
            spatial_w = image_shape[1] // down_factor
            self._spatial_h = spatial_h
            self._spatial_w = spatial_w
            flat_dim = self.encoder.out_channels * spatial_h * spatial_w
            self.latent_down = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.latent_up = nn.Sequential(
                nn.Linear(latent_dim, flat_dim),
                nn.LeakyReLU(0.2, inplace=True),
            )
            bottleneck_channels = self.encoder.out_channels
        else:
            self.latent_conv = nn.Sequential(
                nn.Conv2d(
                    self.encoder.out_channels,
                    latent_dim,
                    kernel_size=1,
                    stride=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.latent_inv = nn.Sequential(
                nn.Conv2d(
                    latent_dim,
                    self.encoder.out_channels,
                    kernel_size=1,
                    stride=1,
                ),
                nn.LeakyReLU(0.2, inplace=True),
            )
            bottleneck_channels = self.encoder.out_channels

        self.decoder = ResidualDecoder(
            out_channels=in_channels,
            base_features=base_features,
            stages=stages,
            bottleneck_channels=bottleneck_channels,
            upsample_position=decoder_upsample_position,
        )

    def _latent_projection(
        self,
        encoded: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dense_bottleneck:
            latent_vec = self.latent_down(encoded)
            inv = self.latent_up(latent_vec)
            inv = inv.view(
                encoded.shape[0],
                encoded.shape[1],
                self._spatial_h,
                self._spatial_w,
            )
            return latent_vec, inv

        latent_map = self.latent_conv(encoded)
        latent_vec = latent_map.flatten(start_dim=1)
        inv = self.latent_inv(latent_map)
        return latent_vec, inv

    def forward(
        self,
        x_noisy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate reconstruction and latent consistency pair from noisy input."""
        encoded = self.encoder(x_noisy)
        z, decoder_input = self._latent_projection(encoded)
        x_rebuilt = self.decoder(decoder_input)

        encoded_rebuilt = self.encoder_reconstructed(x_rebuilt)
        if self._dense_bottleneck:
            z_rebuilt = self.latent_down(encoded_rebuilt)
        else:
            z_rebuilt = self.latent_conv(encoded_rebuilt).flatten(start_dim=1)

        return z, x_rebuilt, z_rebuilt
