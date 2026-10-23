"""
Layer base classes for NeuraCommand.

All layers inherit from the Layer ABC, providing a consistent interface
for forward/backward passes and parameter access.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Iterator, Literal

import numpy as np

from neura_command.backend import get_backend, normalize_device


DeviceType = Literal["cpu", "cuda", "gpu"]


class Layer(ABC):
    """Abstract base class for all layers.

    Layers are the building blocks of neural networks. Each layer:
    - Transforms input during forward pass
    - Computes gradients during backward pass
    - Optionally contains trainable parameters

    Example:
        class Dense(Layer):
            def forward(self, x, training=True):
                return x @ self.weights + self.bias
    """

    _training: bool = True
    _device: DeviceType = "cpu"

    @property
    def training(self) -> bool:
        """Whether the layer is in training mode."""
        return self._training

    @property
    def device(self) -> DeviceType:
        """Device this layer is on ('cpu' or 'cuda')."""
        return self._device

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through the layer.

        Args:
            x: Input array of shape (batch_size, input_features)
            training: Whether in training mode (affects Dropout, BatchNorm, etc.)

        Returns:
            Output array of shape (batch_size, output_features)
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through the layer.

        Args:
            grad: Gradient of loss with respect to layer output.
                  Shape: (batch_size, output_features)

        Returns:
            Gradient of loss with respect to layer input.
            Shape: (batch_size, input_features)
        """
        pass

    def parameters(self) -> List[np.ndarray]:
        """Return list of trainable parameter arrays.

        Override in layers with trainable parameters (Dense, BatchNorm, etc.)

        Returns:
            List of parameter arrays (weights, biases, etc.)
        """
        return []

    def gradients(self) -> List[np.ndarray]:
        """Return list of gradient arrays corresponding to parameters().

        Must be in the same order as parameters().

        Returns:
            List of gradient arrays
        """
        return []

    def parameters_and_gradients(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over (parameter, gradient) pairs.

        Yields:
            Tuples of (parameter_array, gradient_array)
        """
        return zip(self.parameters(), self.gradients())

    def train_mode(self) -> None:
        """Set layer to training mode."""
        self._training = True

    def eval_mode(self) -> None:
        """Set layer to evaluation/inference mode."""
        self._training = False

    def to(self, device: DeviceType) -> "Layer":
        """Move layer to specified device.

        Override in subclasses to move parameters.

        Args:
            device: Target device ('cpu' or 'cuda')

        Returns:
            self (for chaining)
        """
        device = normalize_device(device)
        get_backend(device)
        self._device = device
        return self

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.size for p in self.parameters())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ParameterMixin:
    """Mixin for layers with trainable parameters.

    Provides common parameter initialization strategies.
    """

    @staticmethod
    def xavier_init(shape: Tuple[int, ...], device: DeviceType = "cpu") -> np.ndarray:
        """Xavier/Glorot initialization for weights.

        Good for tanh, sigmoid activations.
        """
        xp = get_backend(device)
        fan_in, fan_out = shape[0], shape[1] if len(shape) > 1 else 1
        limit = float(np.sqrt(6 / (fan_in + fan_out)))
        return xp.random.uniform(-limit, limit, shape).astype(xp.float32)

    @staticmethod
    def he_init(shape: Tuple[int, ...], device: DeviceType = "cpu") -> np.ndarray:
        """He initialization for weights.

        Good for ReLU activations.
        """
        xp = get_backend(device)
        fan_in = shape[0]
        std = float(np.sqrt(2 / fan_in))
        return (xp.random.randn(*shape) * std).astype(xp.float32)

    @staticmethod
    def zeros_init(shape: Tuple[int, ...], device: DeviceType = "cpu") -> np.ndarray:
        """Zero initialization (typically for biases)."""
        xp = get_backend(device)
        return xp.zeros(shape, dtype=xp.float32)

    @staticmethod
    def ones_init(shape: Tuple[int, ...], device: DeviceType = "cpu") -> np.ndarray:
        """Ones initialization."""
        xp = get_backend(device)
        return xp.ones(shape, dtype=xp.float32)
