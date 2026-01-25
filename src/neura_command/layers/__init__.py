"""
NeuraCommand Layers Module

Provides all layer types for building neural networks.
"""

from neura_command.layers.base import Layer, ParameterMixin
from neura_command.layers.dense import Dense
from neura_command.layers.dropout import Dropout
from neura_command.layers.batchnorm import BatchNorm
from neura_command.layers.flatten import Flatten, Input
from neura_command.layers.activation import (
    Activation,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
)

__all__ = [
    # Base
    'Layer',
    'ParameterMixin',
    # Core layers
    'Dense',
    'Dropout',
    'BatchNorm',
    'Flatten',
    'Input',
    # Activations as layers
    'Activation',
    'ReLU',
    'Sigmoid',
    'Tanh',
    'Softmax',
    'LeakyReLU',
    'ELU',
]
