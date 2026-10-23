from abc import ABC, abstractmethod
from math import prod

import numpy as np

from neura_command.backend import get_array_module


class LossFunction(ABC):
    @staticmethod
    def _check_shapes(y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Targets and predictions must have the same shape; got {y_true.shape} and {y_pred.shape}"
            )

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss."""
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Per-sample derivative; parameter layers apply the mean over the batch."""
        pass


class DifferenceLoss(LossFunction):
    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a - b

    def derivative(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        xp = get_array_module(b)
        return xp.ones_like(a - b)


class MSELoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._check_shapes(y_true, y_pred)
        xp = get_array_module(y_pred)
        return xp.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        self._check_shapes(y_true, y_pred)
        return 2 * (y_pred - y_true) / prod(y_pred.shape[1:])


class CrossEntropyLoss(LossFunction):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._check_shapes(y_true, y_pred)
        xp = get_array_module(y_pred)
        epsilon = 1e-15  # Avoid division by zero
        y_pred = xp.clip(y_pred, epsilon, 1 - epsilon)
        return -xp.mean(xp.sum(y_true * xp.log(y_pred), axis=-1))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        self._check_shapes(y_true, y_pred)
        xp = get_array_module(y_pred)
        epsilon = 1e-15  # Avoid division by zero
        y_pred = xp.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred)


class HuberLoss(LossFunction):
    def __init__(self, delta: float = 1.0):
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be finite and positive")
        self.delta = delta

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        self._check_shapes(y_true, y_pred)
        xp = get_array_module(y_pred)
        error = xp.abs(y_true - y_pred)
        quadratic = xp.minimum(error, self.delta)
        linear = error - quadratic
        return xp.mean(0.5 * quadratic**2 + self.delta * linear)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        self._check_shapes(y_true, y_pred)
        xp = get_array_module(y_pred)
        error = y_pred - y_true
        return xp.where(
            xp.abs(error) <= self.delta, error, self.delta * xp.sign(error)
        ) / prod(y_pred.shape[1:])
