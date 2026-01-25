from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class NeuronInterface(ABC):
    @abstractmethod
    def initialize_weights(self):
        pass

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def update_weights(self):
        pass

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int):
        pass
