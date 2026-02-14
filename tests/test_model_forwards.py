import torch
from torch import nn

from grdnet.models.pytorch.blocks import ResidualBlock, ResidualDecoder, ResidualEncoder
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


def test_discriminator_head_outputs_logits_not_probabilities() -> None:
    model = Discriminator(
        in_channels=1,
        base_features=8,
        stages=(1, 1),
        image_shape=(16, 16),
    )
    assert all(
        not isinstance(module, nn.Sigmoid) for module in model.classifier.modules()
    )

    linear = model.classifier[1]
    if not isinstance(linear, nn.Linear):
        raise AssertionError("Expected Linear layer at discriminator head")
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.fill_(2.5)

    x = torch.rand((2, 1, 16, 16), dtype=torch.float32)
    _, logits = model(x)
    assert torch.all(logits > 1.0)


def test_segmentator_forward_shape() -> None:
    model = UNetSegmentator(in_channels=2, base_features=8)
    x = torch.rand((3, 2, 16, 16), dtype=torch.float32)
    y = model(x)
    assert y.shape == (3, 1, 16, 16)


def test_encoder_downsample_position_is_configurable() -> None:
    encoder_first = ResidualEncoder(
        in_channels=1,
        base_features=8,
        stages=(3,),
        downsample_position="first",
    )
    encoder_last = ResidualEncoder(
        in_channels=1,
        base_features=8,
        stages=(3,),
        downsample_position="last",
    )

    first_strides = [block.conv1.stride for block in encoder_first.blocks]
    last_strides = [block.conv1.stride for block in encoder_last.blocks]

    assert first_strides == [(2, 2), (1, 1), (1, 1)]
    assert last_strides == [(1, 1), (1, 1), (2, 2)]


def test_decoder_upsample_position_is_configurable() -> None:
    decoder_first = ResidualDecoder(
        out_channels=1,
        base_features=8,
        stages=(3,),
        bottleneck_channels=8,
        upsample_position="first",
    )
    decoder_last = ResidualDecoder(
        out_channels=1,
        base_features=8,
        stages=(3,),
        bottleneck_channels=8,
        upsample_position="last",
    )

    first_blocks = list(decoder_first.blocks.children())
    last_blocks = list(decoder_last.blocks.children())

    assert isinstance(first_blocks[0], nn.Sequential)
    assert isinstance(first_blocks[1], ResidualBlock)
    assert isinstance(first_blocks[2], ResidualBlock)

    assert isinstance(last_blocks[0], ResidualBlock)
    assert isinstance(last_blocks[1], ResidualBlock)
    assert isinstance(last_blocks[2], nn.Sequential)
