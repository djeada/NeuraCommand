"""
Tests for new layer system.
"""

import numpy as np
import pytest

from neura_command.layers import (
    Layer,
    Dense,
    Dropout,
    BatchNorm,
    Flatten,
    ReLU,
    Sigmoid,
    Softmax,
    Activation,
)
from neura_command.network_utils.activation_functions import ReLUActivation


class TestDenseLayer:
    def test_init(self):
        layer = Dense(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weights.shape == (10, 5)
        assert layer.bias.shape == (1, 5)

    def test_forward_shape(self):
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        assert output.shape == (32, 5)

    def test_forward_with_activation(self):
        layer = Dense(10, 5, activation=ReLUActivation())
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        assert output.shape == (32, 5)
        assert np.all(output >= 0)  # ReLU outputs are non-negative

    def test_backward_shape(self):
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)
        layer.forward(x)
        grad = np.random.randn(32, 5)
        grad_input = layer.backward(grad)
        assert grad_input.shape == (32, 10)

    def test_parameters(self):
        layer = Dense(10, 5)
        params = layer.parameters()
        assert len(params) == 2  # weights and bias
        assert params[0].shape == (10, 5)
        assert params[1].shape == (1, 5)

    def test_gradients_computed(self):
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)
        layer.forward(x)
        grad = np.random.randn(32, 5)
        layer.backward(grad)
        grads = layer.gradients()
        assert len(grads) == 2
        assert grads[0].shape == (10, 5)
        assert grads[1].shape == (1, 5)

    def test_no_bias(self):
        layer = Dense(10, 5, use_bias=False)
        assert layer.bias is None
        assert len(layer.parameters()) == 1

    def test_repr(self):
        layer = Dense(10, 5, activation=ReLUActivation())
        r = repr(layer)
        assert "Dense" in r
        assert "10" in r
        assert "5" in r


class TestDropoutLayer:
    def test_init(self):
        layer = Dropout(0.5)
        assert layer.rate == 0.5

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            Dropout(1.0)
        with pytest.raises(ValueError):
            Dropout(-0.1)

    def test_forward_training(self):
        np.random.seed(42)
        layer = Dropout(0.5)
        x = np.ones((100, 100))
        output = layer.forward(x, training=True)
        # Some values should be zero
        assert np.any(output == 0)
        # Non-zero values should be scaled
        assert np.any(output == 2.0)

    def test_forward_inference(self):
        layer = Dropout(0.5)
        x = np.ones((10, 10))
        output = layer.forward(x, training=False)
        np.testing.assert_array_equal(output, x)

    def test_backward(self):
        np.random.seed(42)
        layer = Dropout(0.5)
        x = np.ones((10, 10))
        layer.forward(x, training=True)
        grad = np.ones((10, 10))
        grad_input = layer.backward(grad)
        assert grad_input.shape == (10, 10)


class TestBatchNormLayer:
    def test_init(self):
        layer = BatchNorm(64)
        assert layer.num_features == 64
        assert layer.gamma.shape == (1, 64)
        assert layer.beta.shape == (1, 64)

    def test_forward_shape(self):
        layer = BatchNorm(64)
        x = np.random.randn(32, 64)
        output = layer.forward(x)
        assert output.shape == (32, 64)

    def test_forward_normalizes(self):
        layer = BatchNorm(64)
        x = np.random.randn(32, 64) * 10 + 5  # Shifted and scaled
        output = layer.forward(x, training=True)
        # Output should be approximately normalized
        assert np.abs(np.mean(output)) < 1
        assert 0.5 < np.std(output) < 2

    def test_backward_shape(self):
        layer = BatchNorm(64)
        x = np.random.randn(32, 64)
        layer.forward(x)
        grad = np.random.randn(32, 64)
        grad_input = layer.backward(grad)
        assert grad_input.shape == (32, 64)

    def test_parameters(self):
        layer = BatchNorm(64)
        params = layer.parameters()
        assert len(params) == 2  # gamma and beta


class TestFlattenLayer:
    def test_forward_2d(self):
        layer = Flatten()
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        assert output.shape == (32, 10)

    def test_forward_4d(self):
        layer = Flatten()
        x = np.random.randn(32, 8, 8, 3)  # Like image batch
        output = layer.forward(x)
        assert output.shape == (32, 8 * 8 * 3)

    def test_backward(self):
        layer = Flatten()
        x = np.random.randn(32, 8, 8, 3)
        layer.forward(x)
        grad = np.random.randn(32, 8 * 8 * 3)
        grad_input = layer.backward(grad)
        assert grad_input.shape == (32, 8, 8, 3)


class TestActivationLayers:
    def test_relu(self):
        layer = ReLU()
        x = np.array([[-1, 0, 1, 2]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, [[0, 0, 1, 2]])

    def test_sigmoid(self):
        layer = Sigmoid()
        x = np.array([[0]])
        output = layer.forward(x)
        np.testing.assert_almost_equal(output, [[0.5]])

    def test_softmax(self):
        layer = Softmax()
        x = np.array([[1, 2, 3]])
        output = layer.forward(x)
        assert output.shape == (1, 3)
        np.testing.assert_almost_equal(np.sum(output), 1.0)

    def test_activation_from_string(self):
        layer = Activation("relu")
        x = np.array([[-1, 1]])
        output = layer.forward(x)
        np.testing.assert_array_equal(output, [[0, 1]])

    def test_activation_unknown(self):
        with pytest.raises(ValueError):
            Activation("unknown_activation")

    def test_backward(self):
        layer = ReLU()
        x = np.array([[-1, 0, 1, 2]], dtype=float)
        layer.forward(x)
        grad = np.ones((1, 4))
        grad_input = layer.backward(grad)
        np.testing.assert_array_equal(grad_input, [[0, 0, 1, 1]])
