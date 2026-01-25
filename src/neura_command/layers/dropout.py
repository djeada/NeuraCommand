"""
Dropout layer for regularization.
"""

from typing import List, Literal

import numpy as np

from neura_command.layers.base import Layer, DeviceType
from neura_command.backend import get_backend, get_default_device


class Dropout(Layer):
    """Dropout layer for regularization.

    Randomly sets a fraction of input units to 0 during training,
    which helps prevent overfitting. Uses inverted dropout (scales
    by 1/(1-rate) during training) so no scaling needed at inference.

    Args:
        rate: Fraction of input units to drop (0 to 1).
        device: Device to create layer on ('cpu' or 'cuda')

    Example:
        model = Sequential([
            Dense(784, 256, activation=ReLU()),
            Dropout(0.5),
            Dense(256, 10, activation=Softmax()),
        ])
    """

    def __init__(self, rate: float = 0.5, device: DeviceType = None):
        if not 0 <= rate < 1:
            raise ValueError(f"Dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        self._mask = None
        self._device = device if device is not None else get_default_device()

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout during forward pass.

        During training: randomly zero out units and scale remaining.
        During inference: pass through unchanged.
        """
        xp = get_backend(self._device)
        self._training = training

        if training and self.rate > 0:
            # Inverted dropout: scale by 1/(1-rate) during training
            self._mask = (xp.random.rand(*x.shape) > self.rate).astype(xp.float32) / (
                1 - self.rate
            )
            return x * self._mask

        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through dropout.

        Gradient only flows through non-dropped units.
        """
        if self._training and self.rate > 0 and self._mask is not None:
            return grad * self._mask
        return grad

    def to(self, device: DeviceType) -> "Dropout":
        """Move layer to specified device."""
        self._device = device
        self._mask = None
        return self

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate}, device='{self._device}')"
