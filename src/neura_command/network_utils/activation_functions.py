from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Apply the activation function to input x."""
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the activation function with respect to input x."""
        pass


class LinearActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class SigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sigmoid = self.compute(x)
        return sigmoid * (1 - sigmoid)


class BasicSigmoidActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)


class ReLUActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)


class StepActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1, 0)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class TanhActivation(ActivationFunction):
    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class SoftmaxActivation(ActivationFunction):
    """Softmax activation function for multi-class classification output layers."""

    def compute(self, x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        # For softmax, the derivative is typically handled differently in backprop
        # when combined with cross-entropy loss. This returns the Jacobian diagonal.
        s = self.compute(x)
        return s * (1 - s)


class LeakyReLUActivation(ActivationFunction):
    """Leaky ReLU activation function."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha)


class ELUActivation(ActivationFunction):
    """Exponential Linear Unit activation function."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def compute(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, self.alpha * np.exp(x))
