"""
Tests for backend (CPU/GPU) abstraction.
"""

import numpy as np
import pytest

from neura_command import (
    Sequential,
    Dense,
    ReLU,
    Dropout,
    BatchNorm,
    get_backend,
    set_default_device,
    get_default_device,
    is_cupy_available,
    to_device,
    to_numpy,
)


class TestBackendBasics:
    def test_get_backend_cpu(self):
        xp = get_backend("cpu")
        assert xp is np

    def test_get_default_device(self):
        # Default should be 'cpu' if we haven't changed it
        device = get_default_device()
        assert device in ("cpu", "cuda")

    def test_is_cupy_available_returns_bool(self):
        result = is_cupy_available()
        assert isinstance(result, bool)

    def test_to_numpy_passthrough(self):
        x = np.array([1, 2, 3])
        result = to_numpy(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(x, result)

    def test_to_device_cpu(self):
        x = np.array([1.0, 2.0, 3.0])
        result = to_device(x, "cpu")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(x, result)


class TestSequentialDevice:
    def test_default_device_cpu(self):
        model = Sequential([Dense(10, 5)])
        assert model.device == "cpu"

    def test_explicit_cpu_device(self):
        model = Sequential([Dense(10, 5)], device="cpu")
        assert model.device == "cpu"

    def test_layers_have_device(self):
        model = Sequential([Dense(10, 5), ReLU()])
        for layer in model.layers:
            assert layer.device == "cpu"

    def test_model_to_cpu(self):
        model = Sequential([Dense(10, 5)])
        model.to("cpu")
        assert model.device == "cpu"

    def test_add_layer_inherits_device(self):
        model = Sequential([], device="cpu")
        model.add(Dense(10, 5))
        assert model.layers[0].device == "cpu"

    def test_training_on_cpu(self):
        np.random.seed(42)
        model = Sequential([Dense(4, 8), ReLU(), Dense(8, 2)], device="cpu")

        X = np.random.randn(20, 4).astype(np.float32)
        y = np.random.randn(20, 2).astype(np.float32)

        # Should train without error
        model.train(X, y, epochs=3, batch_size=8, verbose=False)

        # Should predict without error
        pred = model.predict(X)
        assert pred.shape == (20, 2)
        assert isinstance(pred, np.ndarray)


class TestDenseDevice:
    def test_dense_cpu_weights(self):
        layer = Dense(10, 5, device="cpu")
        assert isinstance(layer.weights, np.ndarray)
        assert isinstance(layer.bias, np.ndarray)

    def test_dense_to_cpu(self):
        layer = Dense(10, 5)
        layer.to("cpu")
        assert layer.device == "cpu"
        assert isinstance(layer.weights, np.ndarray)


class TestDropoutDevice:
    def test_dropout_cpu(self):
        layer = Dropout(0.5, device="cpu")
        assert layer.device == "cpu"

        x = np.random.randn(10, 5).astype(np.float32)
        out = layer.forward(x, training=True)
        assert out.shape == x.shape


class TestBatchNormDevice:
    def test_batchnorm_cpu(self):
        layer = BatchNorm(5, device="cpu")
        assert layer.device == "cpu"
        assert isinstance(layer.gamma, np.ndarray)
        assert isinstance(layer.beta, np.ndarray)

    def test_batchnorm_forward_cpu(self):
        layer = BatchNorm(5, device="cpu")
        x = np.random.randn(10, 5).astype(np.float32)
        out = layer.forward(x, training=True)
        assert out.shape == x.shape


class TestGPUDevice:
    """Tests for GPU functionality. Skipped if CuPy not available."""

    @pytest.fixture(autouse=True)
    def skip_if_no_gpu(self):
        if not is_cupy_available():
            pytest.skip("CuPy not available")

    def test_get_backend_cuda(self):
        xp = get_backend("cuda")
        import cupy as cp

        assert xp is cp

    def test_dense_gpu(self):
        layer = Dense(10, 5, device="cuda")
        import cupy as cp

        assert isinstance(layer.weights, cp.ndarray)

    def test_to_device_gpu(self):
        x = np.array([1.0, 2.0, 3.0])
        x_gpu = to_device(x, "cuda")
        import cupy as cp

        assert isinstance(x_gpu, cp.ndarray)

    def test_model_to_cuda(self):
        model = Sequential([Dense(10, 5)], device="cpu")
        model.to("cuda")
        assert model.device == "cuda"
        import cupy as cp

        assert isinstance(model.layers[0].weights, cp.ndarray)

    def test_training_on_gpu(self):
        model = Sequential([Dense(4, 8), ReLU(), Dense(8, 2)], device="cuda")

        X = np.random.randn(20, 4).astype(np.float32)
        y = np.random.randn(20, 2).astype(np.float32)

        model.train(X, y, epochs=3, batch_size=8, verbose=False)

        # Predictions should come back as numpy
        pred = model.predict(X)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (20, 2)

    def test_gpu_predictions_match_cpu(self):
        np.random.seed(42)

        # Create CPU model
        model_cpu = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])

        # Create GPU model with same weights
        model_gpu = Sequential([Dense(10, 5), ReLU(), Dense(5, 2)])

        # Copy weights from CPU to GPU model
        for cpu_layer, gpu_layer in zip(model_cpu.layers, model_gpu.layers):
            if hasattr(cpu_layer, "weights"):
                gpu_layer.weights = cpu_layer.weights.copy()
                if cpu_layer.use_bias:
                    gpu_layer.bias = cpu_layer.bias.copy()

        model_gpu.to("cuda")

        X = np.random.randn(20, 10).astype(np.float32)

        pred_cpu = model_cpu.predict(X)
        pred_gpu = model_gpu.predict(X)

        np.testing.assert_array_almost_equal(pred_cpu, pred_gpu, decimal=5)
