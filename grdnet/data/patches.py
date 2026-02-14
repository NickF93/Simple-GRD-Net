"""Patch extraction utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def extract_patches(
    images: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
) -> torch.Tensor:
    """Extract fixed-size patches from a BCHW tensor.

    Returns
    -------
    torch.Tensor
        Shape is ``[B * N, C, patch_size, patch_size]``.
    """
    if images.ndim != 4:
        raise ValueError("images must be BCHW tensor")

    bsz, channels, _, _ = images.shape
    unfolded = functional.unfold(images, kernel_size=patch_size, stride=stride)
    n_patches = unfolded.shape[-1]
    patches = unfolded.transpose(1, 2).reshape(
        bsz * n_patches,
        channels,
        patch_size,
        patch_size,
    )
    return patches.contiguous()
