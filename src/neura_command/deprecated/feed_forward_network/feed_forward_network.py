import numpy as np
from typing import List, Optional, Callable

from neura_command.layer.layer import Layer
from neura_command.network_utils.loss_functions import LossFunction

import logging

# Configure the logging system
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class FeedForwardNetwork:
    def __init__(self, loss_func: LossFunction, layers=None) -> None:
        self.layers: List[Layer] = layers if layers is not None else list()
        self.loss_func: LossFunction = loss_func
        logger.info(
            "FeedForwardNetwork initialized with loss function: %s",
            self.loss_func.__class__.__name__,
        )

    @property
    def input_size(self) -> int:
        if self.layers:
            return self.layers[0].input_size
        else:
            return 0

    @property
    def output_size(self) -> int:
        if self.layers:
            return self.layers[-1].output_size
        else:
            return 0

    def add_layer(self, layer: Layer, index: Optional[int] = None) -> None:
        if index is not None:
            self._validate_index(index, allow_equal=True)
            self._validate_layer_sizes(layer, index)
            self.layers.insert(index, layer)
            logger.info("Layer inserted at index %s: %s", index, layer)
        else:
            if self.layers and self.layers[-1].output_size != layer.input_size:
                raise ValueError("Layer sizes are not compatible.")
            self.layers.append(layer)
            logger.info("Layer added: %s", layer)

    def remove_layer(self, index: int) -> None:
        self._validate_index(index)
        del self.layers[index]
        logger.info("Layer removed at index: %s", index)

    def reorder_layers(self, new_order: List[int]) -> None:
        if len(new_order) != len(self.layers):
            raise ValueError("New order must include all layers.")
        if set(new_order) != set(range(len(self.layers))):
            raise ValueError("Invalid layer indices in new order.")

        self.layers = [self.layers[i] for i in new_order]
        self._validate_consecutive_layer_sizes()

    def _validate_index(self, index: int, allow_equal: bool = False) -> None:
        if not (0 <= index < len(self.layers)) and not (
            allow_equal and index == len(self.layers)
        ):
            raise IndexError("Index out of range.")

    def _validate_layer_sizes(self, layer: Layer, index: int) -> None:
        if index > 0 and self.layers[index - 1].output_size != layer.input_size:
            raise ValueError("Layer sizes are not compatible before the index.")
        if (
            index < len(self.layers)
            and self.layers[index].input_size != layer.output_size
        ):
            raise ValueError("Layer sizes are not compatible after the index.")

    def _validate_consecutive_layer_sizes(self) -> None:
        for i in range(len(self.layers) - 1):
            if self.layers[i].output_size != self.layers[i + 1].input_size:
                raise ValueError("Reordered layers are not compatible.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        error = self.loss_func.derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            error = layer.backward(error)[0]
            error = error @ layer.weights[:-1].T  # Exclude the bias weights

    def update_weights(self) -> None:
        for layer in self.layers:
            layer.update_weights()

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.forward(X)
        return np.mean(self.loss_func.compute(y, predictions))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        learning_rate_schedule: Optional[Callable[[int], float]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        logger.info(
            "Starting training with %d epochs, batch size: %d", epochs, batch_size
        )

        for epoch in range(epochs):
            if learning_rate_schedule:
                new_learning_rate = learning_rate_schedule(epoch)
                for layer in self.layers:
                    layer.learning_rate = new_learning_rate
                logger.debug(
                    "Adjusted learning rate to %s at epoch %s", new_learning_rate, epoch
                )

            epoch_loss = self._train_batch(X, y, batch_size)
            logger.info(f"Epoch {epoch + 1}/{epochs}, Training Loss: {epoch_loss}")

            if X_val is not None and y_val is not None:
                val_loss = self.calculate_loss(X_val, y_val)
                logger.info(f"Validation Loss: {val_loss}")

    def _train_batch(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> float:
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            output = self.forward(X_batch)
            self.backward(y_batch, output)
            self.update_weights()
            epoch_loss += np.mean(self.loss_func.compute(y_batch, output))

        return epoch_loss / (len(X) / batch_size)

    def __repr__(self) -> str:
        layers_repr = ", ".join(repr(layer) for layer in self.layers)
        return f"{self.__class__.__name__}(loss_func={self.loss_func.__class__.__name__}, layers=[{layers_repr}])"
