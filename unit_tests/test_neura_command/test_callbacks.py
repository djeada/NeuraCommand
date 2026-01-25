import numpy as np
import pytest

from neura_command.network_utils.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    HistoryCallback,
    CosineAnnealingScheduler,
    ExponentialDecayScheduler,
    StepDecayScheduler,
    ReduceLROnPlateau,
)
from neura_command.network_utils.optimizers import SGD, Adam


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


class TestCosineAnnealingScheduler:
    def test_init(self):
        optimizer = Adam(learning_rate=0.001)
        scheduler = CosineAnnealingScheduler(optimizer, T_max=100, min_lr=1e-6)
        assert scheduler.T_max == 100
        assert scheduler.min_lr == 1e-6

    def test_cosine_decay(self):
        optimizer = Adam(learning_rate=0.1)
        scheduler = CosineAnnealingScheduler(optimizer, T_max=100, min_lr=0.0)
        scheduler.on_train_begin()

        # At epoch 0, should be at initial lr
        scheduler.on_epoch_begin(0)
        assert abs(optimizer.learning_rate - 0.1) < 1e-6

        # At epoch 50 (half cycle), should be around 0.05
        scheduler.on_epoch_begin(50)
        assert abs(optimizer.learning_rate - 0.05) < 0.01

        # At epoch 100, should be at min_lr
        scheduler.on_epoch_begin(100)
        assert abs(optimizer.learning_rate - 0.0) < 1e-6

    def test_warmup(self):
        optimizer = Adam(learning_rate=0.1)
        scheduler = CosineAnnealingScheduler(optimizer, T_max=100, warmup_epochs=10)
        scheduler.on_train_begin()

        # At epoch 0 (first warmup epoch), should be 10% of initial lr
        scheduler.on_epoch_begin(0)
        assert abs(optimizer.learning_rate - 0.01) < 1e-6

        # At epoch 9 (last warmup epoch), should be near initial lr
        scheduler.on_epoch_begin(9)
        assert abs(optimizer.learning_rate - 0.1) < 0.01


class TestExponentialDecayScheduler:
    def test_init(self):
        optimizer = SGD(learning_rate=0.1)
        scheduler = ExponentialDecayScheduler(optimizer, decay_rate=0.95)
        assert scheduler.decay_rate == 0.95

    def test_exponential_decay(self):
        optimizer = SGD(learning_rate=0.1)
        scheduler = ExponentialDecayScheduler(optimizer, decay_rate=0.9)
        scheduler.on_train_begin()

        # Epoch 0: no decay yet
        scheduler.on_epoch_begin(0)
        assert abs(optimizer.learning_rate - 0.1) < 1e-6

        # Epoch 1: first decay
        scheduler.on_epoch_begin(1)
        assert abs(optimizer.learning_rate - 0.09) < 1e-6

        # Epoch 2: second decay
        scheduler.on_epoch_begin(2)
        assert abs(optimizer.learning_rate - 0.081) < 1e-6

    def test_min_lr(self):
        optimizer = SGD(learning_rate=0.1)
        scheduler = ExponentialDecayScheduler(optimizer, decay_rate=0.1, min_lr=0.05)
        scheduler.on_train_begin()

        # After several epochs, should hit min_lr
        for i in range(10):
            scheduler.on_epoch_begin(i)

        assert optimizer.learning_rate >= 0.05


class TestStepDecayScheduler:
    def test_init(self):
        optimizer = SGD(learning_rate=0.1)
        scheduler = StepDecayScheduler(optimizer, step_size=30, gamma=0.1)
        assert scheduler.step_size == 30
        assert scheduler.gamma == 0.1

    def test_step_decay(self):
        optimizer = SGD(learning_rate=0.1)
        scheduler = StepDecayScheduler(optimizer, step_size=10, gamma=0.1)
        scheduler.on_train_begin()

        # Before first step
        scheduler.on_epoch_begin(5)
        assert abs(optimizer.learning_rate - 0.1) < 1e-6

        # After first step (epoch 10)
        scheduler.on_epoch_begin(10)
        assert abs(optimizer.learning_rate - 0.01) < 1e-6

        # After second step (epoch 20)
        scheduler.on_epoch_begin(20)
        assert abs(optimizer.learning_rate - 0.001) < 1e-6


class TestReduceLROnPlateau:
    def test_init(self):
        optimizer = Adam(learning_rate=0.001)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        assert scheduler.patience == 5
        assert scheduler.factor == 0.5

    def test_reduce_on_plateau(self):
        optimizer = Adam(learning_rate=0.1)
        scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        scheduler.on_train_begin()

        # Improving loss
        scheduler.on_epoch_end(0, {"val_loss": 1.0})
        assert optimizer.learning_rate == 0.1

        # No improvement for 2 epochs
        scheduler.on_epoch_end(1, {"val_loss": 1.1})
        scheduler.on_epoch_end(2, {"val_loss": 1.2})

        # Should reduce lr
        assert abs(optimizer.learning_rate - 0.05) < 1e-6

    def test_improvement_resets_patience(self):
        optimizer = Adam(learning_rate=0.1)
        scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        scheduler.on_train_begin()

        scheduler.on_epoch_end(0, {"val_loss": 1.0})
        scheduler.on_epoch_end(1, {"val_loss": 1.1})  # wait = 1
        scheduler.on_epoch_end(2, {"val_loss": 0.9})  # improvement, wait = 0
        scheduler.on_epoch_end(3, {"val_loss": 1.0})  # wait = 1

        # LR should not have changed yet
        assert optimizer.learning_rate == 0.1

    def test_min_lr(self):
        optimizer = Adam(learning_rate=0.1)
        scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.1, min_lr=0.05)
        scheduler.on_train_begin()

        scheduler.on_epoch_end(0, {"val_loss": 1.0})
        scheduler.on_epoch_end(1, {"val_loss": 1.1})  # triggers reduction

        # Should not go below min_lr
        assert optimizer.learning_rate >= 0.05
