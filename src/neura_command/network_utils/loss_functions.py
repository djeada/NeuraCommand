from abc import ABC, abstractmethod

import numpy as np


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
        return np.ones_like(a - b)


class MSELoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true)


class CrossEntropyLoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15  # Avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15  # Avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred)


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = np.abs(y_true - y_pred)
        quadratic = np.minimum(error, self.delta)
        linear = error - quadratic
        return np.mean(0.5 * quadratic ** 2 + self.delta * linear)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error = y_pred - y_true
        delta_mask = np.where(np.abs(error) <= self.delta, 1, 0)
        gradient = delta_mask * error + (1 - delta_mask) * self.delta * np.sign(error)
        return gradient
