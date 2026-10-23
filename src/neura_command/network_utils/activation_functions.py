from abc import ABC, abstractmethod

import numpy as np

from neura_command.backend import get_array_module


class ActivationFunction(ABC):
    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function to input x."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the activation function with respect to input x."""
        pass

    def backward(self, x, grad):
        """Apply the activation's vector-Jacobian product."""
        return grad * self.derivative(x)


class LinearActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.ones_like(x)


class SigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        z = xp.exp(-xp.abs(x))
        return xp.where(x >= 0, 1 / (1 + z), z / (1 + z))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = self.compute(x)
        return sigmoid * (1 - sigmoid)


class BasicSigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        z = xp.exp(-xp.abs(x))
        return xp.where(x >= 0, 1 / (1 + z), z / (1 + z))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)


class ReLUActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(x.dtype)


class StepActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.where(x >= 0, 1, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.ones_like(x)


class TanhActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return 1 - xp.tanh(x) ** 2


class SoftmaxActivation(ActivationFunction):
    """Softmax activation function for multi-class classification output layers."""

    def compute(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        xp = get_array_module(x)
        exp_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
        return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # For softmax, the derivative is typically handled differently in backprop
        # when combined with cross-entropy loss. This returns the Jacobian diagonal.
        s = self.compute(x)
        return s * (1 - s)

    def backward(self, x, grad):
        xp = get_array_module(x)
        probabilities = self.compute(x)
        return probabilities * (
            grad - xp.sum(grad * probabilities, axis=-1, keepdims=True)
        )


class LeakyReLUActivation(ActivationFunction):
    """Leaky ReLU activation function."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.where(x > 0, 1.0, self.alpha).astype(x.dtype, copy=False)


class ELUActivation(ActivationFunction):
    """Exponential Linear Unit activation function."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def compute(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.where(x > 0, x, self.alpha * xp.expm1(xp.minimum(x, 0)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        xp = get_array_module(x)
        return xp.where(x > 0, 1.0, self.alpha * xp.exp(xp.minimum(x, 0)))
