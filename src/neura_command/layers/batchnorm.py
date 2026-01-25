"""
Batch Normalization layer.
"""

from typing import List, Optional, Literal

import numpy as np

from neura_command.layers.base import Layer, ParameterMixin, DeviceType
from neura_command.backend import get_backend, get_default_device, to_device


class BatchNorm(Layer, ParameterMixin):
    """Batch Normalization layer.

    Normalizes inputs to have zero mean and unit variance, then applies
    learnable scale (gamma) and shift (beta) parameters.

    During training: uses batch statistics
    During inference: uses running statistics accumulated during training

    Args:
        num_features: Number of features (channels)
        epsilon: Small constant for numerical stability
        momentum: Momentum for running statistics update
        device: Device to create layer on ('cpu' or 'cuda')

    Reference:
        Ioffe & Szegedy, 2015: "Batch Normalization: Accelerating Deep
        Network Training by Reducing Internal Covariate Shift"

    Example:
        model = Sequential([
            Dense(784, 256),
            BatchNorm(256),
            ReLU(),
            Dense(256, 10),
        ])
    """

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        device: DeviceType = None,
    ):
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        # Use default device if not specified
        self._device = device if device is not None else get_default_device()
        xp = get_backend(self._device)

        # Learnable parameters
        self.gamma = xp.ones((1, num_features), dtype=xp.float32)  # Scale
        self.beta = xp.zeros((1, num_features), dtype=xp.float32)  # Shift

        # Running statistics for inference
        self.running_mean = xp.zeros((1, num_features), dtype=xp.float32)
        self.running_var = xp.ones((1, num_features), dtype=xp.float32)

        # Gradients
        self._grad_gamma: Optional[np.ndarray] = None
        self._grad_beta: Optional[np.ndarray] = None

        # Cache for backward pass
        self._input_normalized: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._input_centered: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with batch normalization.

        Args:
            x: Input of shape (batch_size, num_features)
            training: Whether in training mode

        Returns:
            Normalized output of shape (batch_size, num_features)
        """
        xp = get_backend(self._device)
        self._training = training

        if training:
            # Compute batch statistics
            batch_mean = xp.mean(x, axis=0, keepdims=True)
            batch_var = xp.var(x, axis=0, keepdims=True)

            # Update running statistics
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * batch_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * batch_var

            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics for inference
            mean = self.running_mean
            var = self.running_var

        # Normalize
        self._std = xp.sqrt(var + self.epsilon)
        self._input_centered = x - mean
        self._input_normalized = self._input_centered / self._std

        # Scale and shift
        return self.gamma * self._input_normalized + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through batch normalization.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient to pass to previous layer
        """
        xp = get_backend(self._device)
        batch_size = grad.shape[0]

        # Gradients w.r.t. gamma and beta
        self._grad_gamma = xp.sum(grad * self._input_normalized, axis=0, keepdims=True)
        self._grad_beta = xp.sum(grad, axis=0, keepdims=True)

        # Gradient w.r.t. normalized input
        grad_normalized = grad * self.gamma

        # Gradient w.r.t. input (using batch norm backward formula)
        grad_var = xp.sum(
            grad_normalized * self._input_centered * -0.5 * (self._std**-3),
            axis=0,
            keepdims=True,
        )
        grad_mean = xp.sum(
            grad_normalized * -1 / self._std, axis=0, keepdims=True
        ) + grad_var * xp.mean(-2 * self._input_centered, axis=0, keepdims=True)

        grad_input = (
            grad_normalized / self._std
            + grad_var * 2 * self._input_centered / batch_size
            + grad_mean / batch_size
        )

        return grad_input

    def parameters(self) -> List[np.ndarray]:
        """Return trainable parameters."""
        return [self.gamma, self.beta]

    def gradients(self) -> List[np.ndarray]:
        """Return gradients for parameters."""
        return [self._grad_gamma, self._grad_beta]

    def to(self, device: DeviceType) -> "BatchNorm":
        """Move layer to specified device."""
        if device == self._device:
            return self

        self._device = device
        self.gamma = to_device(self.gamma, device)
        self.beta = to_device(self.beta, device)
        self.running_mean = to_device(self.running_mean, device)
        self.running_var = to_device(self.running_var, device)

        # Clear caches
        self._input_normalized = None
        self._std = None
        self._input_centered = None
        self._grad_gamma = None
        self._grad_beta = None

        return self

    def __repr__(self) -> str:
        return f"BatchNorm(num_features={self.num_features}, device='{self._device}')"
