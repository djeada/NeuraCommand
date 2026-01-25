from typing import List, Tuple

import numpy as np

from neura_command.neuron.perceptron.perceptron import Perceptron


class Layer:
    def __init__(self, perceptrons: List[Perceptron]):
        self.perceptrons = perceptrons

    @property
    def input_size(self) -> int:
        if self.perceptrons:
            return self.perceptrons[0].input_size
        else:
            return 0

    @property
    def output_size(self) -> int:
        return sum(perceptron.output_size for perceptron in self.perceptrons)

    @property
    def weights(self):
        # Concatenating the weights of all perceptrons vertically
        return np.vstack([perceptron.weights for perceptron in self.perceptrons])

    def forward(self, X: np.ndarray) -> np.ndarray:
        outputs = [perceptron.forward(X) for perceptron in self.perceptrons]
        return np.hstack(outputs)

    def backward(self, error: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        total_delta = None
        total_weight_gradient = None

        error_index = 0
        for perceptron in self.perceptrons:
            # Extract the portion of the error corresponding to each perceptron's output
            perceptron_error = error[
                :, error_index : error_index + perceptron.output_size
            ]

            # Perform the backward operation on each perceptron
            delta, weight_gradient = perceptron.backward(perceptron_error)

            # Aggregate deltas and weight gradients
            if total_delta is None:
                total_delta = delta
                total_weight_gradient = weight_gradient
            else:
                total_delta += delta
                total_weight_gradient += weight_gradient

            # Increment the error_index to move to the next perceptron's output segment
            error_index += perceptron.output_size

        return total_delta, total_weight_gradient

    def update_weights(self):
        for perceptron in self.perceptrons:
            perceptron.update_weights()

    def __repr__(self):
        return f"Layer(perceptrons=[{', '.join(repr(p) for p in self.perceptrons)}])"
