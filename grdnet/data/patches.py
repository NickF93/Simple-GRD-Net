"""Patch extraction utilities."""

from __future__ import annotations

import torch
from torch.nn import functional


def extract_patches(
    images: torch.Tensor,
    *,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> torch.Tensor:
    """Extract fixed-size patches from a BCHW tensor.

    Returns
    -------
    torch.Tensor
        Shape is ``[B * N, C, patch_h, patch_w]``.
    """
    if images.ndim != 4:
        raise ValueError("images must be BCHW tensor")

    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    bsz, channels, _, _ = images.shape
    unfolded = functional.unfold(
        images,
        kernel_size=(patch_h, patch_w),
        stride=(stride_h, stride_w),
    )
    n_patches = unfolded.shape[-1]
    patches = unfolded.transpose(1, 2).reshape(
        bsz * n_patches,
        channels,
        patch_h,
        patch_w,
    )
    return patches.contiguous()
