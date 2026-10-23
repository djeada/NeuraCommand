"""
Sequential neural network model.

The main high-level API for building and training neural networks.
"""

from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from neura_command.layers.base import Layer, DeviceType
from neura_command.network_utils.optimizers import Optimizer, Adam
from neura_command.network_utils.loss_functions import LossFunction, MSELoss
from neura_command.network_utils.callbacks import Callback, HistoryCallback
from neura_command.backend import (
    get_backend,
    get_default_device,
    to_device,
    to_numpy,
    normalize_device,
    get_array_module,
)

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class Sequential:
    """Sequential neural network model.

    A linear stack of layers that can be trained end-to-end.

    Args:
        layers: List of Layer objects
        optimizer: Optimizer instance (default: Adam)
        loss: Loss function instance (default: MSELoss)
        device: Device to use ('cpu' or 'cuda')

    Example:
        from neura_command.layers import Dense, Dropout, ReLU, Softmax
        from neura_command.network_utils.optimizers import Adam
        from neura_command.network_utils.loss_functions import CrossEntropyLoss

        model = Sequential(
            layers=[
                Dense(784, 256),
                ReLU(),
                Dropout(0.2),
                Dense(256, 128),
                ReLU(),
                Dropout(0.2),
                Dense(128, 10),
                Softmax(),
            ],
            optimizer=Adam(learning_rate=0.001),
            loss=CrossEntropyLoss(),
            device='cuda',  # Use GPU
        )

        model.train(X_train, y_train, epochs=10, batch_size=32)
        predictions = model.predict(X_test)
    """

    def __init__(
        self,
        layers: Optional[List[Layer]] = None,
        optimizer: Optional[Optimizer] = None,
        loss: Optional[LossFunction] = None,
        device: DeviceType = None,
    ):
        self.layers: List[Layer] = layers if layers is not None else []
        self.optimizer: Optimizer = optimizer if optimizer is not None else Adam()
        self.loss: LossFunction = loss if loss is not None else MSELoss()
        self._training: bool = True
        self._device: DeviceType = normalize_device(
            device if device is not None else get_default_device()
        )
        self.to(self._device)

    @property
    def device(self) -> DeviceType:
        """Get the device this model is on."""
        return self._device

    def to(self, device: DeviceType) -> "Sequential":
        """Move model to specified device.

        Args:
            device: Target device ('cpu' or 'cuda')

        Returns:
            self (for chaining)

        Example:
            model = Sequential([Dense(10, 5)])
            model.to('cuda')  # Move to GPU
        """
        device = normalize_device(device)
        get_backend(device)  # Validate before changing model state.
        old_params = [param for layer in self.layers for param in layer.parameters()]
        for layer in self.layers:
            layer.to(device)
        new_params = [param for layer in self.layers for param in layer.parameters()]
        self.optimizer.to(
            device, {id(old): id(new) for old, new in zip(old_params, new_params)}
        )
        self._device = device
        return self

    def add(self, layer: Layer) -> "Sequential":
        """Add a layer to the model.

        Args:
            layer: Layer instance to add

        Returns:
            self (for chaining)
        """
        layer.to(self._device)
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass through all layers.

        Args:
            x: Input data of shape (batch_size, input_features)
            training: Whether in training mode

        Returns:
            Output of shape (batch_size, output_features)
        """
        # Move input to device if needed
        x = to_device(x, self._device)
        x = x.astype(get_backend(self._device).float32, copy=False)

        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        """Backward pass through all layers.

        Args:
            grad: Gradient of loss w.r.t. network output
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions (inference mode).

        Args:
            x: Input data

        Returns:
            Model predictions (always returned as NumPy array on CPU)
        """
        output = self.forward(x, training=False)
        return to_numpy(output)

    def train_mode(self) -> None:
        """Set all layers to training mode."""
        self._training = True
        for layer in self.layers:
            layer.train_mode()

    def eval_mode(self) -> None:
        """Set all layers to evaluation mode."""
        self._training = False
        for layer in self.layers:
            layer.eval_mode()

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss value."""
        return self.loss.compute(y_true, y_pred)

    def compute_loss_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions."""
        return self.loss.derivative(y_true, y_pred)

    @staticmethod
    def _loss_to_float(loss_value: Any) -> float:
        """Convert backend scalar loss to a Python float."""
        if hasattr(loss_value, "get"):
            return float(to_numpy(loss_value))
        return float(loss_value)

    def _average_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
    ) -> float:
        """Compute sample-weighted average loss over a dataset."""
        was_training = self._training
        self.eval_mode()

        n_samples = len(X)
        total_loss = 0.0

        for i in range(0, n_samples, batch_size):
            X_batch = X[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            y_pred = self.forward(X_batch, training=False)
            batch_loss = self.compute_loss(
                to_device(y_batch, self._device).astype(y_pred.dtype, copy=False),
                y_pred,
            )
            total_loss += batch_loss * len(X_batch)

        if was_training:
            self.train_mode()

        return self._loss_to_float(total_loss / n_samples)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        callbacks: Optional[List[Callback]] = None,
        shuffle: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the model.

        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            validation_data: Optional (X_val, y_val) tuple for validation
            callbacks: List of Callback objects
            shuffle: Whether to shuffle data each epoch
            verbose: Whether to print progress

        Returns:
            Dictionary containing training history
        """
        self._validate_data(X, y, batch_size)
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if validation_data is not None:
            self._validate_data(*validation_data, batch_size)
        self.train_mode()

        # Setup callbacks
        callbacks = list(callbacks or [])
        history = HistoryCallback()
        callbacks.append(history)

        # Notify callbacks of training start
        for cb in callbacks:
            cb.on_train_begin()

        n_samples = len(X)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # Notify callbacks
            for cb in callbacks:
                cb.on_epoch_begin(epoch)

            # Shuffle data
            if shuffle:
                indices = np.random.permutation(n_samples)
            else:
                indices = None

            epoch_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)

                selection = slice(start, end) if indices is None else indices[start:end]
                # Only gather a batch; avoid a second full dataset each epoch.
                X_batch = X[
                    selection
                    if get_array_module(X) is np or indices is None
                    else get_array_module(X).asarray(selection)
                ]
                y_batch = y[
                    selection
                    if get_array_module(y) is np or indices is None
                    else get_array_module(y).asarray(selection)
                ]

                # Forward pass (automatically moves X to device)
                y_pred = self.forward(X_batch, training=True)

                # Move y_batch to device for loss computation
                y_batch_device = to_device(y_batch, self._device).astype(
                    y_pred.dtype, copy=False
                )

                # Compute loss
                batch_loss = self.compute_loss(y_batch_device, y_pred)
                epoch_loss += batch_loss * len(X_batch)

                # Backward pass
                grad = self.compute_loss_gradient(y_batch_device, y_pred)
                self.backward(grad)

                # Update weights
                self.optimizer.step(self.layers)

            epoch_loss = self._loss_to_float(epoch_loss / n_samples)

            # Compute validation loss
            logs = {"loss": epoch_loss, "epoch": epoch}

            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self._average_loss(X_val, y_val, batch_size=batch_size)
                logs["val_loss"] = val_loss

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f}"
                if "val_loss" in logs:
                    msg += f" - val_loss: {logs['val_loss']:.4f}"
                logger.info(msg)

            # Notify callbacks
            stop_training = False
            for cb in callbacks:
                if cb.on_epoch_end(epoch, logs):
                    stop_training = True

            if stop_training:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Notify callbacks of training end
        for cb in callbacks:
            cb.on_train_end()

        return history.get_history()

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """Evaluate the model on test data.

        Args:
            X: Test input data
            y: Test target data
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with 'loss' and optionally other metrics
        """
        self._validate_data(X, y, batch_size)
        self.eval_mode()
        return {"loss": self._average_loss(X, y, batch_size=batch_size)}

    @staticmethod
    def _validate_data(X, y, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(X) == 0 or len(X) != len(y):
            raise ValueError(
                "Features and targets must have matching, non-empty batches"
            )

    def summary(self) -> str:
        """Return a summary of the model architecture.

        Returns:
            String description of the model
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Model Summary")
        lines.append("=" * 60)

        total_params = 0
        for i, layer in enumerate(self.layers):
            layer_params = layer.num_parameters
            total_params += layer_params
            lines.append(f"Layer {i}: {layer!r}")
            if layer_params > 0:
                lines.append(f"         Parameters: {layer_params:,}")

        lines.append("=" * 60)
        lines.append(f"Total parameters: {total_params:,}")
        lines.append(f"Optimizer: {self.optimizer.__class__.__name__}")
        lines.append(f"Loss: {self.loss.__class__.__name__}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, path: str, **kwargs) -> None:
        """Save model to disk.

        Creates two files:
        - {path}.json: Model architecture
        - {path}.npz: Model weights

        Args:
            path: Base path (without extension)
            **kwargs: Additional arguments passed to save_model()

        Example:
            model.save('models/my_model')
        """
        from neura_command.serialization import save_model

        save_model(self, path, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs) -> "Sequential":
        """Load model from disk.

        Args:
            path: Base path (without extension)
            **kwargs: Additional arguments passed to load_model()

        Returns:
            Loaded Sequential model

        Example:
            model = Sequential.load('models/my_model')
        """
        from neura_command.serialization import load_model

        return load_model(path, **kwargs)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(layer.num_parameters for layer in self.layers)

    def __repr__(self) -> str:
        return f"Sequential(layers={len(self.layers)}, params={self.num_parameters:,})"
