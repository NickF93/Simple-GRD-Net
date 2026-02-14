import torch

from grdnet.models.pytorch.discriminator import Discriminator
from grdnet.models.pytorch.generator import GeneratorEDE
from grdnet.models.pytorch.segmentator import UNetSegmentator


def test_generator_forward_dense_false_shapes() -> None:
    model = GeneratorEDE(
        in_channels=1,
        base_features=8,
        stages=(1, 1),
        latent_dim=4,
        dense_bottleneck=False,
        image_shape=(16, 16),
    )
    x = torch.rand((2, 1, 16, 16), dtype=torch.float32)
    z, x_rebuilt, z_rebuilt = model(x)
    assert z.shape[0] == 2
    assert x_rebuilt.shape == x.shape
    assert z_rebuilt.shape == z.shape


def test_generator_forward_dense_true_shapes() -> None:
    model = GeneratorEDE(
        in_channels=1,
        base_features=8,
        stages=(1, 1),
        latent_dim=6,
        dense_bottleneck=True,
        image_shape=(16, 16),
    )
    x = torch.rand((1, 1, 16, 16), dtype=torch.float32)
    z, x_rebuilt, z_rebuilt = model(x)
    assert z.shape == (1, 6)
    assert x_rebuilt.shape == x.shape
    assert z_rebuilt.shape == z.shape


def test_discriminator_forward_shapes() -> None:
    model = Discriminator(
        in_channels=1,
        base_features=8,
        stages=(1, 1),
        image_shape=(16, 16),
    )
    x = torch.rand((2, 1, 16, 16), dtype=torch.float32)
    features, probs = model(x)
    assert features.shape[0] == 2
    assert probs.shape == (2, 1)


def test_segmentator_forward_shape() -> None:
    model = UNetSegmentator(in_channels=2, base_features=8)
    x = torch.rand((3, 2, 16, 16), dtype=torch.float32)
    y = model(x)
    assert y.shape == (3, 1, 16, 16)
