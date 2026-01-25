from typing import Tuple

import numpy as np


class Dropout:
    """Dropout layer for regularization.
    
    Randomly sets a fraction of input units to 0 during training,
    which helps prevent overfitting.
    """

    def __init__(self, rate: float = 0.5):
        """
        Initialize Dropout layer.
        
        Args:
            rate: Fraction of the input units to drop (0 to 1).
        """
        if not 0 <= rate < 1:
            raise ValueError("Dropout rate must be in [0, 1)")
        self.rate = rate
        self.mask: np.ndarray = None
        self.training: bool = True

    @property
    def input_size(self) -> int:
        return 0  # Dropout doesn't change dimensions

    @property
    def output_size(self) -> int:
        return 0  # Dropout doesn't change dimensions

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply dropout during forward pass."""
        if self.training and self.rate > 0:
            # Create mask and scale by 1/(1-rate) for inverted dropout
            self.mask = (np.random.rand(*X.shape) > self.rate) / (1 - self.rate)
            return X * self.mask
        return X

    def backward(self, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate error through dropout layer."""
        if self.training and self.rate > 0 and self.mask is not None:
            return error * self.mask, np.zeros((1, 1))
        return error, np.zeros((1, 1))

    def update_weights(self) -> None:
        """Dropout has no weights to update."""
        pass

    def train_mode(self) -> None:
        """Set layer to training mode."""
        self.training = True

    def eval_mode(self) -> None:
        """Set layer to evaluation mode (no dropout applied)."""
        self.training = False

    def __repr__(self) -> str:
        return f"Dropout(rate={self.rate})"
