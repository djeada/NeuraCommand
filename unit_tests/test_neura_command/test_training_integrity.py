"""Numerical gradient checks and observable training/callback behavior."""

import numpy as np
import pytest

from neura_command import (
    Sequential,
    Dense,
    BatchNorm,
    Tanh,
    Softmax,
    Dropout,
    SGD,
    MSELoss,
    HuberLoss,
    CrossEntropyLoss,
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    set_random_seed,
)


@pytest.mark.parametrize("loss", [MSELoss(), HuberLoss(0.7), CrossEntropyLoss()])
@pytest.mark.parametrize("normalization", [False, True])
def test_full_network_gradients_match_mean_loss_finite_differences(loss, normalization):
    set_random_seed(19)
    rng = np.random.default_rng(19)
    layers = [Dense(3, 4)]
    if normalization:
        layers.append(BatchNorm(4))
    layers.extend([Tanh(), Dense(4, 2)])
    if isinstance(loss, CrossEntropyLoss):
        layers.append(Softmax())
    # Float64 parameters isolate derivative correctness from float32 roundoff.
    for layer in layers:
        for name in ("weights", "bias", "gamma", "beta", "running_mean", "running_var"):
            array = getattr(layer, name, None)
            if array is not None:
                setattr(layer, name, array.astype(np.float64))
    model = Sequential(layers, optimizer=SGD(0), loss=loss)
    x = rng.normal(size=(5, 3)).astype(np.float32)
    y = rng.normal(size=(5, 2))
    if isinstance(loss, CrossEntropyLoss):
        y = np.eye(2)[[0, 1, 1, 0, 1]]
    prediction = model.forward(x, training=True)
    model.backward(loss.derivative(y, prediction))
    analytic = [
        (parameter, gradient.copy())
        for layer in layers
        for parameter, gradient in layer.parameters_and_gradients()
    ]
    epsilon = 1e-5
    for parameter, expected in analytic:
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            high = loss.compute(y, model.forward(x, training=True))
            parameter[index] = original - epsilon
            low = loss.compute(y, model.forward(x, training=True))
            parameter[index] = original
            numerical[index] = (high - low) / (2 * epsilon)
        np.testing.assert_allclose(expected, numerical, atol=1e-7, rtol=1e-5)


def test_batch_duplication_does_not_change_parameter_update():
    set_random_seed(4)
    first = Sequential(
        [Dense(2, 4), BatchNorm(4), Tanh(), Dense(4, 2)], optimizer=SGD(0.03)
    )
    second = Sequential(
        [Dense(2, 4), BatchNorm(4), Tanh(), Dense(4, 2)], optimizer=SGD(0.03)
    )
    second.set_weights(first.get_weights())
    x = np.array([[0.2, 0.7], [0.6, -0.3], [-1, 2]], np.float32)
    y = np.array([[0, 1], [0.5, -0.2], [1, 0]], np.float32)
    first.train(x, y, epochs=1, batch_size=3, shuffle=False, verbose=False)
    second.train(
        np.tile(x, (2, 1)),
        np.tile(y, (2, 1)),
        epochs=1,
        batch_size=6,
        shuffle=False,
        verbose=False,
    )
    for actual, expected in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_one_dimensional_numeric_data_is_a_single_feature_batch():
    model = Sequential([Dense(1, 1)], optimizer=SGD(0.02))
    model.layers[0].weights[:] = 0
    model.layers[0].bias[:] = 0
    history = model.train([1, 2, 3], [2, 4, 6], epochs=30, verbose=False)
    assert history["loss"][-1] < history["loss"][0]
    assert model.evaluate([1, 2, 3], [2, 4, 6])["loss"] < 0.1


@pytest.mark.parametrize("bad", [[], [np.nan, 2], [np.inf, 2], ["x", "y"], 4])
def test_invalid_data_fails_before_parameter_updates(bad):
    model = Sequential([Dense(1, 1)])
    before = model.get_weights()
    with pytest.raises(ValueError):
        model.train(bad, [1, 2], verbose=False)
    for actual, expected in zip(model.get_weights(), before):
        np.testing.assert_array_equal(actual, expected)


def test_targets_cannot_silently_broadcast_across_output_features():
    model = Sequential([Dense(2, 2)])
    with pytest.raises(ValueError, match="same shape"):
        model.train(np.ones((4, 2)), np.ones(4), verbose=False)


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_core_rejects_non_integer_or_empty_batch_sizes(value):
    model = Sequential([Dense(2, 1)])
    with pytest.raises(ValueError, match="positive integer"):
        model.train(np.ones((4, 2)), np.ones((4, 1)), batch_size=value, verbose=False)


def test_scheduler_updates_optimizer_and_history_records_actual_rate():
    model = Sequential([Dense(1, 1, use_bias=False)], optimizer=SGD(1.0))
    model.layers[0].weights[:] = 0
    callback = LearningRateScheduler(lambda epoch, rate: rate / 2)
    history = model.train(
        np.ones((2, 1)), np.ones((2, 1)), epochs=3, callbacks=[callback], verbose=False
    )
    np.testing.assert_allclose(history["learning_rate"], [0.5, 0.25, 0.125])
    np.testing.assert_allclose(model.layers[0].weights, 1.0)


def test_early_stopping_restores_parameters_and_batchnorm_buffers():
    class ScriptedEpoch(Callback):
        def on_epoch_end(self, epoch, logs=None):
            for array in self.model._weight_arrays():
                array[...] = epoch + 1
            logs["loss"] = epoch  # First epoch is best; second triggers stopping.

    model = Sequential([Dense(2, 3), BatchNorm(3), Dense(3, 1)])
    stopping = EarlyStopping(monitor="loss", patience=1)
    history = model.train(
        np.ones((4, 2)),
        np.ones((4, 1)),
        epochs=5,
        callbacks=[ScriptedEpoch(), stopping],
        verbose=False,
    )
    assert len(history["loss"]) == 2
    assert stopping.best_epoch == 0
    assert stopping.stopped_epoch == 1
    for value in model.get_weights():
        np.testing.assert_array_equal(value, np.ones_like(value))
    assert model.optimizer._t == 0
    assert model.optimizer._m == {}


def test_batch_hooks_receive_batch_sizes_and_numeric_losses():
    class Recorder(Callback):
        def __init__(self):
            self.starts, self.ends = [], []

        def on_batch_begin(self, batch, logs=None):
            self.starts.append((batch, logs["size"]))

        def on_batch_end(self, batch, logs=None):
            self.ends.append((batch, logs["size"], logs["loss"]))

    recorder = Recorder()
    Sequential([Dense(1, 1)]).train(
        [[1], [2], [3]],
        [[1], [2], [3]],
        epochs=1,
        batch_size=2,
        callbacks=[recorder],
        verbose=False,
    )
    assert recorder.starts == [(0, 2), (1, 1)]
    assert [(batch, size) for batch, size, _ in recorder.ends] == recorder.starts
    assert all(
        isinstance(loss, float) and np.isfinite(loss) for _, _, loss in recorder.ends
    )


def test_seed_reproduces_shuffle_and_dropout_training():
    set_random_seed(10)
    first = Sequential([Dense(2, 8), Tanh(), Dropout(0.3), Dense(8, 1)])
    second = Sequential([Dense(2, 8), Tanh(), Dropout(0.3), Dense(8, 1)])
    second.set_weights(first.get_weights())
    x = np.arange(16, dtype=np.float32).reshape(8, 2) / 10
    y = x[:, :1] * 0.3
    first_history = first.train(x, y, epochs=4, batch_size=3, seed=123, verbose=False)
    second_history = second.train(x, y, epochs=4, batch_size=3, seed=123, verbose=False)
    assert first_history == second_history
    for actual, expected in zip(first.get_weights(), second.get_weights()):
        np.testing.assert_array_equal(actual, expected)


def test_eval_mode_controls_forward_and_evaluation_restores_training_mode():
    model = Sequential([Dense(2, 12), Dropout(0.5)])
    x = np.ones((20, 2), np.float32)
    model.eval_mode()
    np.testing.assert_array_equal(model.forward(x), model.forward(x))
    model.train_mode()
    assert not np.array_equal(model.forward(x), model.forward(x))
    model.evaluate(x, np.ones((20, 12)))
    assert model.training
    assert all(layer.training for layer in model.layers)
    with pytest.raises(ValueError, match="same shape"):
        model.evaluate(x, np.ones((20, 3)))
    assert model.training
    assert all(layer.training for layer in model.layers)


def test_invalid_weight_snapshots_are_atomic_and_good_snapshots_detached():
    model = Sequential([Dense(2, 3), BatchNorm(3)])
    snapshot = model.get_weights()
    model.layers[0].weights[:] = 3
    assert not np.all(snapshot[0] == 3)
    before = model.get_weights()
    bad = [np.zeros_like(value) for value in before]
    bad[-1] = np.zeros((99,))
    with pytest.raises(ValueError, match="shape"):
        model.set_weights(bad)
    for actual, expected in zip(model.get_weights(), before):
        np.testing.assert_array_equal(actual, expected)
    model.set_weights(snapshot)
    for actual, expected in zip(model.get_weights(), snapshot):
        np.testing.assert_array_equal(actual, expected)


def test_float64_inputs_that_overflow_float32_are_rejected_before_training():
    model = Sequential([Dense(1, 1)])
    before = model.get_weights()
    with pytest.raises(ValueError, match="float32"):
        model.train([[1e300]], [[1]], verbose=False)
    for actual, expected in zip(model.get_weights(), before):
        np.testing.assert_array_equal(actual, expected)


def test_non_scalar_loss_fails_before_optimizer_updates():
    from neura_command.network_utils.loss_functions import DifferenceLoss

    model = Sequential([Dense(1, 1)], loss=DifferenceLoss())
    snapshot = model.get_weights()
    with pytest.raises(ValueError, match="scalar"):
        model.train([[1], [2]], [[1], [2]], epochs=1, verbose=False)
    for actual, expected in zip(model.get_weights(), snapshot):
        np.testing.assert_array_equal(actual, expected)
