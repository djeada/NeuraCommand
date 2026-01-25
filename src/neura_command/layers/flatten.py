"""
Utility layers (Flatten, Input, etc.)
"""

import numpy as np

from neura_command.layers.base import Layer


class Flatten(Layer):
    """Flattens input to 1D (preserving batch dimension).

    Useful when transitioning from Conv layers to Dense layers.

    Example:
        model = Sequential([
            # ... conv layers with shape (batch, height, width, channels)
            Flatten(),  # -> (batch, height * width * channels)
            Dense(256, activation=ReLU()),
        ])
    """

    def __init__(self):
        self._input_shape = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Flatten input.

        Args:
            x: Input of shape (batch_size, ...)

        Returns:
            Flattened output of shape (batch_size, product_of_other_dims)
        """
        self._input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Reshape gradient back to original input shape."""
        return grad.reshape(self._input_shape)

    def __repr__(self) -> str:
        return "Flatten()"


class Input(Layer):
    """Input layer (placeholder, mostly for documentation/validation).

    Optional layer that can be used to specify input shape explicitly.
    """

    def __init__(self, shape):
        """
        Args:
            shape: Expected input shape (excluding batch dimension)
        """
        self.shape = shape if isinstance(shape, tuple) else (shape,)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Pass through (validates shape if needed)."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Pass through."""
        return grad

    def __repr__(self) -> str:
        return f"Input(shape={self.shape})"
