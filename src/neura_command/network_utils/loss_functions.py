from abc import ABC, abstractmethod

import numpy as np

from neura_command.backend import get_array_module


class LossFunction(ABC):
    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss."""
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the derivative of the loss."""
        pass


class DifferenceLoss(LossFunction):
    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def derivative(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        xp = get_array_module(b)
        return xp.ones_like(a - b)


class MSELoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        xp = get_array_module(y_pred)
        return xp.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)


class CrossEntropyLoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        xp = get_array_module(y_pred)
        epsilon = 1e-15  # Avoid division by zero
        y_pred = xp.clip(y_pred, epsilon, 1 - epsilon)
        return -xp.mean(xp.sum(y_true * xp.log(y_pred), axis=-1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        xp = get_array_module(y_pred)
        epsilon = 1e-15  # Avoid division by zero
        y_pred = xp.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred)


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        xp = get_array_module(y_pred)
        error = xp.abs(y_true - y_pred)
        quadratic = xp.minimum(error, self.delta)
        linear = error - quadratic
        return xp.mean(0.5 * quadratic**2 + self.delta * linear)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        xp = get_array_module(y_pred)
        error = y_pred - y_true
        return xp.where(xp.abs(error) <= self.delta, error, self.delta * xp.sign(error))
