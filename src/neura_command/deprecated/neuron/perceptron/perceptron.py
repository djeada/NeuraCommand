from typing import Tuple

import numpy as np

from neura_command.network_utils.activation_functions import ActivationFunction
from neura_command.network_utils.loss_functions import DifferenceLoss, LossFunction
from neura_command.neuron.neuron_interface import NeuronInterface


class Perceptron(NeuronInterface):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_func: ActivationFunction,
        loss_func: LossFunction = DifferenceLoss(),
        learning_rate: float = 0.1,
        l1_ratio: float = 0.0,
        l2_ratio: float = 0.0,
    ) -> None:
        self.initialize_weights(input_size, output_size)
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.last_input = np.array([])
        self.last_output = np.array([])
        self.last_gradient = np.array([])

    @property
    def error_func(self) -> LossFunction:
        """Deprecated: Use loss_func instead."""
        return self.loss_func

    @error_func.setter
    def error_func(self, value: LossFunction) -> None:
        """Deprecated: Use loss_func instead."""
        self.loss_func = value

    def initialize_weights(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.normal(
            0, 1 / np.sqrt(input_size), (input_size + 1, output_size)
        )

    @property
    def input_size(self) -> int:
        return self.weights.shape[0] - 1

    @property
    def output_size(self) -> int:
        return self.weights.shape[1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.last_input = np.c_[X, np.ones(X.shape[0])]
        z = self.last_input @ self.weights
        self.last_output = self.activation_func.compute(z)
        return self.last_output

    def backward(self, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        delta = error * self.activation_func.derivative(self.last_output)
        weight_gradient = -self.last_input.T @ delta
        self.last_gradient = weight_gradient
        return delta, weight_gradient

    def update_weights(self):
        l1_term = self.l1_ratio * np.sign(self.weights)
        l2_term = self.l2_ratio * self.weights

        self.weights -= self.learning_rate * (self.last_gradient + l1_term + l2_term)

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int):
        for epoch in range(epochs):
            # Iterating through batches
            for i in range(0, X.shape[0], batch_size):
                # Slicing the batch
                X_batch = X[i : i + batch_size]
                y_batch = y[i : i + batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Calculating error
                error = self.error_func.compute(y_batch, output)

                # Backward pass and weight update
                self.backward(error)
                self.update_weights()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(input_size={self.input_size}, "
            f"output_size={self.output_size}, "
            f"activation_func={self.activation_func.__class__.__name__}, "
            f"loss_func={self.loss_func.__class__.__name__}, "
            f"learning_rate={self.learning_rate}, "
            f"l1_ratio={self.l1_ratio}, "
            f"l2_ratio={self.l2_ratio})"
        )
