from abc import ABC
from typing import Optional, List, Dict, Any

import numpy as np


class Callback(ABC):
    """Base class for training callbacks."""

    model = None

    def set_model(self, model) -> None:
        """Bind the model before training starts."""
        self.model = model

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Called at the end of each epoch. Return True to stop training."""
        return False

    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Example:
        early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)
        network.train(X, y, epochs=100, callbacks=[early_stop])
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialize EarlyStopping callback.

        Args:
            monitor: Metric name to monitor (e.g., 'val_loss', 'loss').
            min_delta: Minimum change to qualify as an improvement.
            patience: Number of epochs with no improvement after which training stops.
            mode: 'min' or 'max'. In 'min' mode, training stops when monitored
                  quantity stops decreasing; in 'max' mode when it stops increasing.
            restore_best_weights: Whether to restore model weights from the epoch
                                  with the best value of the monitored quantity.
        """
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if not np.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and non-negative")
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best: float = np.inf if mode == "min" else -np.inf
        self.best_epoch: int = 0
        self.best_weights: Optional[List[np.ndarray]] = None
        self.wait: int = 0
        self.stopped_epoch: int = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.best = np.inf if self.mode == "min" else -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return False

        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights:
                if self.model is not None:
                    self.best_weights = self.model.get_weights()
                elif "weights" in logs:
                    self.best_weights = [w.copy() for w in logs["weights"]]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True  # Signal to stop training

        return False

    def on_train_end(self, logs=None):
        if (
            self.restore_best_weights
            and self.best_weights is not None
            and self.model is not None
        ):
            # Running statistics are part of the snapshot. Discard moments from
            # later epochs so subsequent training starts from consistent state.
            self.model.set_weights(self.best_weights)

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta

    def get_best_weights(self) -> Optional[List[np.ndarray]]:
        """Return the best weights if restore_best_weights is True."""
        return self.best_weights


class LearningRateScheduler(Callback):
    """Learning rate scheduler callback.

    Example:
        def schedule(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * 0.9

        lr_scheduler = LearningRateScheduler(schedule)
        network.train(X, y, epochs=100, callbacks=[lr_scheduler])
    """

    def __init__(self, schedule):
        """
        Initialize LearningRateScheduler.

        Args:
            schedule: A function that takes an epoch index (int) and current
                      learning rate (float) and returns a new learning rate.
        """
        self.schedule = schedule
        self.current_lr: float = 0.0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current_lr = logs.get("learning_rate", 0.01)
        if self.model is not None:
            current_lr = self.model.optimizer.learning_rate
        new_lr = self.schedule(epoch, current_lr)
        if not np.isfinite(new_lr) or new_lr < 0:
            raise ValueError("Scheduled learning rate must be finite and non-negative")
        self.current_lr = new_lr
        if self.model is not None:
            self.model.optimizer.learning_rate = new_lr
        elif "set_learning_rate" in logs:
            logs["set_learning_rate"](new_lr)


class HistoryCallback(Callback):
    """Records training history."""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.history = {}

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        for key, value in logs.items():
            # Check for numeric types (including numpy scalars)
            try:
                float(value)
                is_numeric = True
            except (TypeError, ValueError):
                is_numeric = False

            if is_numeric:
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(float(value))
        return False

    def get_history(self) -> Dict[str, List[float]]:
        return self.history


class CosineAnnealingScheduler(Callback):
    """Cosine annealing learning rate scheduler.

    Reduces learning rate following a cosine curve from initial_lr to min_lr
    over T_max epochs, then optionally restarts.

    Args:
        optimizer: Optimizer whose learning rate will be adjusted
        T_max: Number of epochs for one cosine cycle
        min_lr: Minimum learning rate (default: 0)
        warmup_epochs: Number of warmup epochs with linear increase (default: 0)

    Reference:
        Loshchilov & Hutter, 2016: "SGDR: Stochastic Gradient Descent with
        Warm Restarts"

    Example:
        optimizer = Adam(learning_rate=0.001)
        scheduler = CosineAnnealingScheduler(optimizer, T_max=50, min_lr=1e-6)
        model.train(X, y, epochs=100, callbacks=[scheduler])
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        min_lr: float = 0.0,
        warmup_epochs: int = 0,
    ):
        self.optimizer = optimizer
        self.T_max = T_max
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.initial_lr: float = optimizer.learning_rate
        self.current_lr: float = self.initial_lr

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.initial_lr = self.optimizer.learning_rate

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_T_max = self.T_max - self.warmup_epochs
            if adjusted_T_max > 0:
                self.current_lr = (
                    self.min_lr
                    + (self.initial_lr - self.min_lr)
                    * (1 + np.cos(np.pi * adjusted_epoch / adjusted_T_max))
                    / 2
                )
            else:
                self.current_lr = self.min_lr

        self.optimizer.learning_rate = self.current_lr


class ExponentialDecayScheduler(Callback):
    """Exponential decay learning rate scheduler.

    Reduces learning rate by a fixed factor at each epoch.

    Args:
        optimizer: Optimizer whose learning rate will be adjusted
        decay_rate: Multiplicative decay factor applied each epoch (e.g., 0.95)
        min_lr: Minimum learning rate (default: 1e-8)
        decay_steps: Apply decay every N epochs (default: 1)

    Example:
        optimizer = SGD(learning_rate=0.1)
        scheduler = ExponentialDecayScheduler(optimizer, decay_rate=0.95)
        model.train(X, y, epochs=100, callbacks=[scheduler])
    """

    def __init__(
        self,
        optimizer,
        decay_rate: float = 0.95,
        min_lr: float = 1e-8,
        decay_steps: int = 1,
    ):
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.min_lr = min_lr
        self.decay_steps = decay_steps
        self.initial_lr: float = optimizer.learning_rate
        self.current_lr: float = self.initial_lr

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.initial_lr = self.optimizer.learning_rate
        self.current_lr = self.initial_lr

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if epoch > 0 and epoch % self.decay_steps == 0:
            self.current_lr = max(
                self.current_lr * self.decay_rate,
                self.min_lr,
            )
            self.optimizer.learning_rate = self.current_lr


class StepDecayScheduler(Callback):
    """Step decay learning rate scheduler.

    Reduces learning rate by a factor at specific epoch milestones.

    Args:
        optimizer: Optimizer whose learning rate will be adjusted
        step_size: Number of epochs between each decay
        gamma: Multiplicative factor for decay (default: 0.1)
        min_lr: Minimum learning rate (default: 1e-8)

    Example:
        optimizer = SGD(learning_rate=0.1)
        # Decay by factor of 0.1 every 30 epochs
        scheduler = StepDecayScheduler(optimizer, step_size=30, gamma=0.1)
        model.train(X, y, epochs=100, callbacks=[scheduler])
    """

    def __init__(
        self,
        optimizer,
        step_size: int,
        gamma: float = 0.1,
        min_lr: float = 1e-8,
    ):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        self.initial_lr: float = optimizer.learning_rate
        self.current_lr: float = self.initial_lr

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.initial_lr = self.optimizer.learning_rate
        self.current_lr = self.initial_lr

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        # Calculate the number of completed step_size intervals
        num_decays = epoch // self.step_size
        new_lr = self.initial_lr * (self.gamma**num_decays)
        self.current_lr = max(new_lr, self.min_lr)
        self.optimizer.learning_rate = self.current_lr


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.

    Args:
        optimizer: Optimizer whose learning rate will be adjusted
        monitor: Metric to monitor (default: 'val_loss')
        factor: Factor by which to reduce learning rate (default: 0.1)
        patience: Number of epochs with no improvement before reducing LR
        min_lr: Minimum learning rate (default: 1e-8)
        mode: 'min' or 'max'. In 'min' mode, LR is reduced when monitored
              value stops decreasing; in 'max' mode when it stops increasing.
        min_delta: Threshold for measuring improvement (default: 1e-4)

    Example:
        optimizer = Adam(learning_rate=0.001)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        model.train(X, y, epochs=100, callbacks=[scheduler], validation_data=(X_val, y_val))
    """

    def __init__(
        self,
        optimizer,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-8,
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        self.optimizer = optimizer
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.min_delta = min_delta

        self.best: float = np.inf if mode == "min" else -np.inf
        self.wait: int = 0
        self.current_lr: float = optimizer.learning_rate

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.best = np.inf if self.mode == "min" else -np.inf
        self.wait = 0
        self.current_lr = self.optimizer.learning_rate

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return False

        if self._is_improvement(current):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                new_lr = max(self.current_lr * self.factor, self.min_lr)
                if new_lr < self.current_lr:
                    self.current_lr = new_lr
                    self.optimizer.learning_rate = new_lr
                self.wait = 0

        return False

    def _is_improvement(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta
