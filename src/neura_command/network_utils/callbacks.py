from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

import numpy as np


class Callback(ABC):
    """Base class for training callbacks."""

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

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            return False

        if self._is_improvement(current):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best_weights and "weights" in logs:
                self.best_weights = [w.copy() for w in logs["weights"]]
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True  # Signal to stop training
        
        return False

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
        new_lr = self.schedule(epoch, current_lr)
        self.current_lr = new_lr
        if "set_learning_rate" in logs:
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
            if isinstance(value, (int, float)):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)
        return False

    def get_history(self) -> Dict[str, List[float]]:
        return self.history
