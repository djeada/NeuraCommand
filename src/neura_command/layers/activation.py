"""
Activation layers for NeuraCommand.

These wrap activation functions as layers, allowing them to be used
in Sequential models as separate components.
"""

import numpy as np

from neura_command.layers.base import Layer, DeviceType
from neura_command.backend import get_default_device, normalize_device, get_backend
from neura_command.network_utils.activation_functions import (
    ActivationFunction,
    ReLUActivation,
    SigmoidActivation,
    TanhActivation,
    SoftmaxActivation,
    LeakyReLUActivation,
    ELUActivation,
    LinearActivation,
)


class Activation(Layer):
    """Activation function as a layer.

    Wraps an ActivationFunction to be used as a standalone layer.

    Args:
        activation: ActivationFunction instance or string name

    Example:
        model = Sequential([
            Dense(784, 128),
            Activation('relu'),
            Dense(128, 10),
            Activation('softmax'),
        ])
    """

    _activation_map = {
        "relu": ReLUActivation,
        "sigmoid": SigmoidActivation,
        "tanh": TanhActivation,
        "softmax": SoftmaxActivation,
        "leaky_relu": LeakyReLUActivation,
        "leakyrelu": LeakyReLUActivation,
        "elu": ELUActivation,
        "linear": LinearActivation,
        "none": LinearActivation,
    }

    def __init__(self, activation, device: DeviceType = None):
        if isinstance(activation, str):
            activation_lower = activation.lower()
            if activation_lower not in self._activation_map:
                raise ValueError(
                    f"Unknown activation: {activation}. "
                    f"Available: {list(self._activation_map.keys())}"
                )
            self.activation = self._activation_map[activation_lower]()
        elif isinstance(activation, ActivationFunction):
            self.activation = activation
        else:
            raise TypeError(
                f"activation must be str or ActivationFunction, got {type(activation)}"
            )

        self._input_cache = None
        self._device = normalize_device(
            device if device is not None else get_default_device()
        )
        get_backend(self._device)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply activation function."""
        self._input_cache = x if training else None
        return self.activation.compute(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through activation."""
        return self.activation.backward(self._input_cache, grad)

    def to(self, device: DeviceType) -> "Activation":
        """Move layer to specified device (no-op for activations)."""
        super().to(device)
        self._input_cache = None
        return self

    def __repr__(self) -> str:
        return f"Activation({self.activation.__class__.__name__})"


# Convenience classes for common activations
class ReLU(Activation):
    """ReLU activation layer."""

    def __init__(self, device: DeviceType = None):
        super().__init__("relu", device=device)


class Sigmoid(Activation):
    """Sigmoid activation layer."""

    def __init__(self, device: DeviceType = None):
        super().__init__("sigmoid", device=device)


class Tanh(Activation):
    """Tanh activation layer."""

    def __init__(self, device: DeviceType = None):
        super().__init__("tanh", device=device)


class Softmax(Activation):
    """Softmax activation layer."""

    def __init__(self, device: DeviceType = None):
        super().__init__("softmax", device=device)


class LeakyReLU(Activation):
    """Leaky ReLU activation layer."""

    def __init__(self, alpha: float = 0.01, device: DeviceType = None):
        self.activation = LeakyReLUActivation(alpha)
        self._input_cache = None
        self._device = normalize_device(
            device if device is not None else get_default_device()
        )
        get_backend(self._device)


class ELU(Activation):
    """ELU activation layer."""

    def __init__(self, alpha: float = 1.0, device: DeviceType = None):
        self.activation = ELUActivation(alpha)
        self._input_cache = None
        self._device = normalize_device(
            device if device is not None else get_default_device()
        )
        get_backend(self._device)
