import numpy as np
import pytest

from neura_command.network_utils.loss_functions import (
    LossFunction,
    DifferenceLoss,
    MSELoss,
    CrossEntropyLoss,
    HuberLoss,
)


class TestDifferenceLoss:
    def test_compute(self):
        loss = DifferenceLoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        result = loss.compute(y_true, y_pred)
        np.testing.assert_array_almost_equal(result, [-0.5, -0.5, 0.5])

    def test_derivative(self):
        loss = DifferenceLoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 2.5])
        result = loss.derivative(y_true, y_pred)
        np.testing.assert_array_equal(result, np.ones(3))


class TestMSELoss:
    def test_compute_zero_error(self):
        loss = MSELoss()
        y = np.array([1.0, 2.0, 3.0])
        result = loss.compute(y, y)
        assert result == 0.0

    def test_compute(self):
        loss = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        result = loss.compute(y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        np.testing.assert_almost_equal(result, expected)

    def test_derivative(self):
        loss = MSELoss()
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        result = loss.derivative(y_true, y_pred)
        expected = 2 * (y_pred - y_true)
        np.testing.assert_array_almost_equal(result, expected)


class TestCrossEntropyLoss:
    def test_compute(self):
        loss = CrossEntropyLoss()
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.9, 0.05, 0.05])
        result = loss.compute(y_true, y_pred)
        assert result > 0

    def test_compute_perfect_prediction(self):
        loss = CrossEntropyLoss()
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([1.0, 0.0, 0.0])
        result = loss.compute(y_true, y_pred)
        # Should be close to 0 (clipped to avoid log(0))
        assert result < 0.001

    def test_derivative_shape(self):
        loss = CrossEntropyLoss()
        y_true = np.array([1.0, 0.0, 0.0])
        y_pred = np.array([0.9, 0.05, 0.05])
        result = loss.derivative(y_true, y_pred)
        assert result.shape == y_true.shape


class TestHuberLoss:
    def test_compute_small_error(self):
        loss = HuberLoss(delta=1.0)
        y_true = np.array([1.0])
        y_pred = np.array([1.5])  # error = 0.5, < delta
        result = loss.compute(y_true, y_pred)
        expected = 0.5 * 0.5**2  # quadratic region
        np.testing.assert_almost_equal(result, expected)

    def test_compute_large_error(self):
        loss = HuberLoss(delta=1.0)
        y_true = np.array([1.0])
        y_pred = np.array([4.0])  # error = 3.0, > delta
        result = loss.compute(y_true, y_pred)
        # Linear region: 0.5 * delta^2 + delta * (error - delta)
        expected = 0.5 * 1.0 + 1.0 * 2.0
        np.testing.assert_almost_equal(result, expected)

    def test_derivative_small_error(self):
        loss = HuberLoss(delta=1.0)
        y_true = np.array([1.0])
        y_pred = np.array([1.5])
        result = loss.derivative(y_true, y_pred)
        expected = np.array([0.5])  # y_pred - y_true in quadratic region
        np.testing.assert_array_almost_equal(result, expected)
