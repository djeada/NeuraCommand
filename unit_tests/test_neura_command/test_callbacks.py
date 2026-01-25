import numpy as np
import pytest

from neura_command.network_utils.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    HistoryCallback,
)


class TestEarlyStopping:
    def test_init(self):
        early_stop = EarlyStopping(patience=5)
        assert early_stop.patience == 5
        assert early_stop.monitor == "val_loss"

    def test_improvement_resets_wait(self):
        early_stop = EarlyStopping(patience=3, mode="min")
        early_stop.on_train_begin()
        
        # Improving loss
        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        assert early_stop.wait == 0
        
        early_stop.on_epoch_end(1, {"val_loss": 0.9})
        assert early_stop.wait == 0
        
        early_stop.on_epoch_end(2, {"val_loss": 0.8})
        assert early_stop.wait == 0

    def test_no_improvement_increments_wait(self):
        early_stop = EarlyStopping(patience=3, mode="min")
        early_stop.on_train_begin()
        
        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        assert early_stop.wait == 0
        
        # No improvement
        early_stop.on_epoch_end(1, {"val_loss": 1.1})
        assert early_stop.wait == 1
        
        early_stop.on_epoch_end(2, {"val_loss": 1.2})
        assert early_stop.wait == 2

    def test_stops_after_patience(self):
        early_stop = EarlyStopping(patience=2, mode="min")
        early_stop.on_train_begin()
        
        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        assert early_stop.on_epoch_end(1, {"val_loss": 1.1}) is False
        assert early_stop.on_epoch_end(2, {"val_loss": 1.2}) is True

    def test_min_delta(self):
        early_stop = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        early_stop.on_train_begin()
        
        early_stop.on_epoch_end(0, {"val_loss": 1.0})
        
        # Improvement less than min_delta doesn't count
        early_stop.on_epoch_end(1, {"val_loss": 0.95})
        assert early_stop.wait == 1  # Not enough improvement

    def test_max_mode(self):
        early_stop = EarlyStopping(patience=2, mode="max")
        early_stop.on_train_begin()
        
        early_stop.on_epoch_end(0, {"val_loss": 0.5})
        assert early_stop.best == 0.5
        
        early_stop.on_epoch_end(1, {"val_loss": 0.6})
        assert early_stop.best == 0.6
        assert early_stop.wait == 0

    def test_missing_metric(self):
        early_stop = EarlyStopping(patience=2)
        early_stop.on_train_begin()
        
        # Should not crash if metric is missing
        result = early_stop.on_epoch_end(0, {})
        assert result is False


class TestLearningRateScheduler:
    def test_schedule_called(self):
        called = []
        
        def schedule(epoch, lr):
            called.append((epoch, lr))
            return lr * 0.9
        
        scheduler = LearningRateScheduler(schedule)
        scheduler.on_epoch_begin(0, {"learning_rate": 0.1})
        
        assert len(called) == 1
        assert called[0] == (0, 0.1)
        assert abs(scheduler.current_lr - 0.09) < 1e-9

    def test_step_decay_schedule(self):
        def step_decay(epoch, lr):
            if epoch > 0 and epoch % 5 == 0:
                return lr * 0.5
            return lr
        
        scheduler = LearningRateScheduler(step_decay)
        
        lr = 0.1
        for epoch in range(11):
            scheduler.on_epoch_begin(epoch, {"learning_rate": lr})
            lr = scheduler.current_lr
        
        # After epochs 5 and 10, lr should be halved twice
        assert abs(lr - 0.025) < 0.001


class TestHistoryCallback:
    def test_records_metrics(self):
        history = HistoryCallback()
        history.on_train_begin()
        
        history.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.1})
        history.on_epoch_end(1, {"loss": 0.9, "val_loss": 1.0})
        history.on_epoch_end(2, {"loss": 0.8, "val_loss": 0.9})
        
        h = history.get_history()
        assert h["loss"] == [1.0, 0.9, 0.8]
        assert h["val_loss"] == [1.1, 1.0, 0.9]

    def test_ignores_non_numeric(self):
        history = HistoryCallback()
        history.on_train_begin()
        
        history.on_epoch_end(0, {"loss": 1.0, "weights": [1, 2, 3]})
        
        h = history.get_history()
        assert "loss" in h
        assert "weights" not in h

    def test_reset_on_train_begin(self):
        history = HistoryCallback()
        history.on_train_begin()
        history.on_epoch_end(0, {"loss": 1.0})
        
        history.on_train_begin()
        assert history.get_history() == {}
