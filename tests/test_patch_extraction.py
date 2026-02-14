import torch

from grdnet.data.patches import extract_patches


def test_extract_patches_shape() -> None:
    images = torch.randn(2, 3, 16, 16)
    patches = extract_patches(images, patch_size=8, stride=8)
    assert patches.shape == (8, 3, 8, 8)
