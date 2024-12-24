from .segmentator_net import build_segmentator
from .generator_discriminator_nets import build_discriminator, build_generator

__all__ = [
    "build_segmentator",
    "build_discriminator",
    "build_generator",
]

__version__ = "0.0.1"
__author__ = "Niccolò Ferrari"
