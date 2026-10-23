"""Device dispatch, portable persistence, and CPU/CUDA training integration."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from neura_command import (
    Sequential,
    Dense,
    ReLU,
    BatchNorm,
    Dropout,
    Softmax,
    Adam,
    SGD,
    RMSprop,
    AdaGrad,
    MSELoss,
    HuberLoss,
    CrossEntropyLoss,
    InferenceModel,
)
from neura_command.backend import DeviceContext, get_default_device
from neura_command.backend import array_backend as backend
from neura_command.network_utils.optimizers import Adadelta, Nadam
from neura_command.network_utils import activation_functions as activations
from neura_command.network_utils import loss_functions as losses
from neura_command.network_utils import optimizers
from neura_command.network_utils.activation_functions import SoftmaxActivation


OPTIMIZERS = [
    lambda: SGD(0.01, momentum=0.9, weight_decay=0.01, l1_regularization=0.001),
    lambda: Adam(0.01, weight_decay=0.01),
    lambda: RMSprop(0.01),
    lambda: AdaGrad(0.01),
    lambda: Adadelta(),
    lambda: Nadam(0.01),
]


class NumpyForbidden:
    """Array math must use the array's namespace, not the imported NumPy module."""

    def __getattr__(self, name):
        raise AssertionError(f"Hard-coded NumPy operation: {name}")


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_optimizers_dispatch_through_array_backend(monkeypatch, factory):
    optimizer = factory()
    monkeypatch.setattr(optimizers, "np", NumpyForbidden())
    weights = np.array([[1, -2]], np.float32)
    gradient = np.array([[0.2, -0.3]], np.float32)
    updated = optimizer.update(id(weights), weights, gradient)
    assert np.isfinite(updated).all()
    assert not np.array_equal(updated, weights)


@pytest.mark.parametrize(
    "activation",
    [
        activations.LinearActivation(),
        activations.SigmoidActivation(),
        activations.BasicSigmoidActivation(),
        activations.ReLUActivation(),
        activations.StepActivation(),
        activations.TanhActivation(),
        activations.SoftmaxActivation(),
        activations.LeakyReLUActivation(),
        activations.ELUActivation(),
    ],
)
def test_activations_dispatch_through_array_backend(monkeypatch, activation):
    monkeypatch.setattr(activations, "np", NumpyForbidden())
    x = np.array([[-1, 0, 1]], np.float32)
    assert np.isfinite(activation.compute(x)).all()
    assert np.isfinite(activation.derivative(x)).all()


@pytest.mark.parametrize("loss", [MSELoss(), HuberLoss(), CrossEntropyLoss()])
def test_losses_dispatch_through_array_backend(monkeypatch, loss):
    monkeypatch.setattr(losses, "np", NumpyForbidden())
    target = np.array([[0, 1]], np.float32)
    prediction = np.array([[0.2, 0.8]], np.float32)
    assert np.isfinite(loss.compute(target, prediction))
    assert np.isfinite(loss.derivative(target, prediction)).all()


def test_zero_cuda_devices_is_unavailable(monkeypatch):
    monkeypatch.setattr(backend, "CUPY_AVAILABLE", True)
    monkeypatch.setattr(
        backend,
        "cp",
        SimpleNamespace(
            cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=lambda: 0))
        ),
    )
    assert not backend.is_cupy_available()
    with pytest.raises(ValueError, match="CUDA"):
        backend.get_backend("cuda")


def test_gpu_transfer_converts_python_sequences(monkeypatch):
    converter = Mock(return_value="gpu array")
    monkeypatch.setattr(backend, "is_cupy_available", lambda: True)
    monkeypatch.setattr(backend, "cp", SimpleNamespace(asarray=converter))
    assert backend.to_device([[1, 2]], "gpu") == "gpu array"
    converter.assert_called_once_with([[1, 2]])


def test_device_context_is_thread_local(monkeypatch):
    monkeypatch.setattr(backend, "is_cupy_available", lambda: True)
    with DeviceContext("cpu"):
        with DeviceContext("gpu"):
            assert get_default_device() == "cuda"
            seen = []
            thread = threading.Thread(target=lambda: seen.append(get_default_device()))
            thread.start()
            thread.join()
            assert seen == ["cpu"]
        assert get_default_device() == "cpu"


def test_failed_device_transfer_leaves_model_usable(monkeypatch):
    model = Sequential([Dense(2, 3), ReLU()])
    expected = model.predict(np.ones((2, 2), np.float32))
    monkeypatch.setattr(backend, "is_cupy_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA"):
        model.to("gpu")
    assert model.device == "cpu"
    assert all(layer.device == "cpu" for layer in model.layers)
    np.testing.assert_array_equal(model.predict(np.ones((2, 2), np.float32)), expected)


def test_optimizer_transfer_remaps_moments_without_resetting_timestep():
    model = Sequential([Dense(2, 3)])
    x = np.ones((4, 2), np.float32)
    model.train(x, np.zeros((4, 3), np.float32), epochs=2, verbose=False)
    optimizer = model.optimizer
    old_id = id(model.layers[0].weights)
    moment = optimizer._m[old_id].copy()
    optimizer.to("cpu", {old_id: 1234})
    assert old_id not in optimizer._m
    np.testing.assert_array_equal(optimizer._m[1234], moment)
    assert optimizer._t == 2


def test_softmax_backward_matches_finite_difference():
    activation = SoftmaxActivation()
    x = np.array([[0.3, -0.4, 0.6]], np.float64)
    grad = np.array([[0.2, 0.7, -0.3]], np.float64)
    analytic = activation.backward(x, grad)
    numerical = np.zeros_like(x)
    epsilon = 1e-6
    for index in range(x.shape[1]):
        high, low = x.copy(), x.copy()
        high[0, index] += epsilon
        low[0, index] -= epsilon
        numerical[0, index] = np.sum(
            (activation.compute(high) - activation.compute(low)) * grad
        ) / (2 * epsilon)
    np.testing.assert_allclose(analytic, numerical, atol=1e-9)


@pytest.mark.parametrize("invalid", [0, -1])
def test_invalid_batches_and_empty_training_data_raise(invalid):
    model = Sequential([Dense(2, 1)])
    x, y = np.ones((4, 2)), np.ones((4, 1))
    with pytest.raises(ValueError, match="batch_size"):
        model.train(x, y, batch_size=invalid)
    with pytest.raises(ValueError, match="non-empty"):
        model.train(x[:0], y[:0])


@pytest.fixture
def cuda():
    if not backend.is_cupy_available():
        pytest.skip("Usable CUDA device and CuPy required")
    import cupy as cp

    return cp


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_cuda_training_validation_and_resume_match_cpu(cuda, tmp_path, factory):
    rng = np.random.default_rng(42)
    cpu = Sequential(
        [Dense(4, 6), BatchNorm(6), ReLU(), Dense(6, 2)], optimizer=factory()
    )
    cpu.save(tmp_path / "initial")
    gpu = Sequential.load(tmp_path / "initial", device="gpu")
    assert gpu.device == "cuda"
    gpu.optimizer = factory()
    x = rng.normal(size=(16, 4)).astype(np.float32)
    y = rng.normal(size=(16, 2)).astype(np.float32)
    cpu_history = cpu.train(
        x,
        y,
        epochs=3,
        batch_size=8,
        shuffle=False,
        verbose=False,
        validation_data=(x, y),
    )
    gpu_history = gpu.train(
        cuda.asarray(x),
        cuda.asarray(y),
        epochs=3,
        batch_size=8,
        shuffle=False,
        verbose=False,
        validation_data=(cuda.asarray(x), cuda.asarray(y)),
    )
    np.testing.assert_allclose(
        gpu_history["loss"], cpu_history["loss"], rtol=1e-4, atol=1e-5
    )
    np.testing.assert_allclose(
        gpu_history["val_loss"], cpu_history["val_loss"], rtol=1e-4, atol=1e-5
    )
    expected = gpu.predict(x)
    gpu.save(tmp_path / "gpu_trained")
    portable = Sequential.load(tmp_path / "gpu_trained", device="cpu")
    np.testing.assert_allclose(portable.predict(x), expected, atol=1e-5)
    gpu.to("cpu")
    gpu.train(x, y, epochs=1, batch_size=8, shuffle=False, verbose=False)
    cpu.train(x, y, epochs=1, batch_size=8, shuffle=False, verbose=False)
    np.testing.assert_allclose(gpu.predict(x), cpu.predict(x), rtol=1e-4, atol=1e-5)
    gpu.to("cuda")
    gpu.train(x, y, epochs=1, verbose=False)
    assert np.isfinite(gpu.predict(x)).all()
    for layer in gpu.layers:
        assert all(isinstance(param, cuda.ndarray) for param in layer.parameters())


@pytest.mark.parametrize("precision", [None, "float16", "int8"])
def test_cuda_inference_export_and_portable_artifacts(cuda, tmp_path, precision):
    model = Sequential(
        [Dense(4, 5, activation=SoftmaxActivation()), Dropout(0.2), Dense(5, 2)],
        device="gpu",
    )
    x = np.ones((4, 4), np.float32)
    exported = InferenceModel.from_sequential(
        model, quantize=precision, optimize=True, per_channel=True, device="cuda"
    )
    expected = exported.predict(cuda.asarray(x))
    assert isinstance(exported.layers[0].weights, cuda.ndarray)
    exported.save(tmp_path / "gpu_export")
    loaded = InferenceModel.load(tmp_path / "gpu_export", device="gpu")
    np.testing.assert_allclose(loaded.predict(x), expected, atol=1e-5)
    loaded.to("cpu")
    np.testing.assert_allclose(loaded.predict(x), expected, atol=1e-5)
    host_export = InferenceModel.load(tmp_path / "gpu_export")
    assert host_export.device == "cpu"
    np.testing.assert_allclose(host_export.predict(x), expected, atol=1e-5)


@pytest.mark.parametrize("loss", [HuberLoss(), CrossEntropyLoss()])
def test_cuda_losses_and_dropout_training(cuda, loss):
    model = Sequential(
        [Dense(4, 6), ReLU(), Dropout(0.2), Dense(6, 2), Softmax()],
        device="cuda",
        loss=loss,
    )
    x = np.ones((8, 4), np.float32)
    y = np.tile([1, 0], (8, 1)).astype(np.float32)
    history = model.train(x, y, epochs=2, verbose=False, validation_data=(x, y))
    assert np.isfinite(history["loss"]).all()
    assert np.isfinite(model.evaluate(cuda.asarray(x), cuda.asarray(y))["loss"])


@pytest.mark.parametrize("factory", OPTIMIZERS)
def test_optimizer_configuration_and_predictions_survive_save_load(tmp_path, factory):
    model = Sequential([Dense(2, 3)], optimizer=factory())
    x = np.ones((3, 2), np.float32)
    model.save(tmp_path / "portable")
    loaded = Sequential.load(tmp_path / "portable")
    assert type(loaded.optimizer) is type(model.optimizer)
    for name, value in vars(model.optimizer).items():
        if isinstance(value, (float, int)):
            assert getattr(loaded.optimizer, name) == value
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))


def test_cuda_callbacks_restore_buffers_and_record_scheduled_rates(cuda):
    from neura_command import EarlyStopping, LearningRateScheduler

    model = Sequential(
        [Dense(2, 4), BatchNorm(4), ReLU(), Dropout(0.2), Dense(4, 1)], device="cuda"
    )
    stopping = EarlyStopping(monitor="loss", patience=1, min_delta=100)
    scheduler = LearningRateScheduler(lambda epoch, rate: rate * 0.8)
    x, y = cuda.ones((8, 2), cuda.float32), cuda.ones((8, 1), cuda.float32)
    history = model.train(
        x,
        y,
        epochs=4,
        batch_size=4,
        callbacks=[scheduler, stopping],
        seed=19,
        verbose=False,
    )
    assert len(history["loss"]) == 2
    np.testing.assert_allclose(history["learning_rate"], [0.0008, 0.00064])
    assert all(isinstance(value, cuda.ndarray) for value in stopping.best_weights)
    for actual, expected in zip(model.get_weights(), stopping.best_weights):
        np.testing.assert_array_equal(cuda.asnumpy(actual), cuda.asnumpy(expected))
    assert model.optimizer._t == 0
    assert np.isfinite(model.predict(x)).all()
