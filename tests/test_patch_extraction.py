import torch

from grdnet.data.patches import extract_patches


def test_extract_patches_shape() -> None:
    images = torch.randn(2, 3, 16, 16)
    patches = extract_patches(images, patch_size=(8, 8), stride=(8, 8))
    assert patches.shape == (8, 3, 8, 8)


def test_extract_patches_rectangular_shape() -> None:
    images = torch.randn(1, 1, 8, 10)
    patches = extract_patches(images, patch_size=(4, 5), stride=(4, 5))
    assert patches.shape == (4, 1, 4, 5)
