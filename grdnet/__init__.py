'''
The GRD-Net module.

This module contains all the classes and functions needed to build the GRD-Net architecture.
'''

from . import util
from . import grd_nets
from . import trainer
from . import perlin
from . import augment
from . import aggregator
from . import data

__all__ = [
    'util',
    'grd_nets',
    'trainer',
    'perlin',
    'augment',
    'aggregator',
    'data',
]

__version__ = '0.0.1'
__author__ = 'Niccolò Ferrari'
