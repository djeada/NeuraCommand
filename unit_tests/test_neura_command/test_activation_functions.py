import numpy as np
import pytest

from neura_command.network_utils.activation_functions import (
    ActivationFunction,
    LinearActivation,
    SigmoidActivation,
    BasicSigmoidActivation,
    ReLUActivation,
    StepActivation,
    TanhActivation,
)


class TestLinearActivation:
    def test_compute(self):
        activation = LinearActivation()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = activation.compute(x)
        np.testing.assert_array_equal(result, x)

    def test_derivative(self):
        activation = LinearActivation()
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = activation.derivative(x)
        np.testing.assert_array_equal(result, np.ones_like(x))


class TestSigmoidActivation:
    def test_compute_bounds(self):
        activation = SigmoidActivation()
        x = np.array([-100.0, 0.0, 100.0])
        result = activation.compute(x)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_compute_zero(self):
        activation = SigmoidActivation()
        result = activation.compute(np.array([0.0]))
        np.testing.assert_almost_equal(result, [0.5])

    def test_derivative_positive(self):
        activation = SigmoidActivation()
        x = np.array([-2.0, 0.0, 2.0])
        result = activation.derivative(x)
        assert np.all(result >= 0)

    def test_derivative_max_at_zero(self):
        activation = SigmoidActivation()
        x = np.array([-2.0, 0.0, 2.0])
        result = activation.derivative(x)
        assert result[1] == max(result)


class TestReLUActivation:
    def test_compute_positive(self):
        activation = ReLUActivation()
        x = np.array([1.0, 2.0, 3.0])
        result = activation.compute(x)
        np.testing.assert_array_equal(result, x)

    def test_compute_negative(self):
        activation = ReLUActivation()
        x = np.array([-1.0, -2.0, -3.0])
        result = activation.compute(x)
        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_compute_mixed(self):
        activation = ReLUActivation()
        x = np.array([-2.0, 0.0, 2.0])
        result = activation.compute(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 2.0])

    def test_derivative(self):
        activation = ReLUActivation()
        x = np.array([-2.0, 0.0, 2.0])
        result = activation.derivative(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])


class TestStepActivation:
    def test_compute(self):
        activation = StepActivation()
        x = np.array([-1.0, 0.0, 1.0])
        result = activation.compute(x)
        np.testing.assert_array_equal(result, [0, 1, 1])


class TestTanhActivation:
    def test_compute_bounds(self):
        activation = TanhActivation()
        x = np.array([-100.0, 0.0, 100.0])
        result = activation.compute(x)
        assert np.all(result >= -1) and np.all(result <= 1)

    def test_compute_zero(self):
        activation = TanhActivation()
        result = activation.compute(np.array([0.0]))
        np.testing.assert_almost_equal(result, [0.0])

    def test_derivative_at_zero(self):
        activation = TanhActivation()
        result = activation.derivative(np.array([0.0]))
        np.testing.assert_almost_equal(result, [1.0])
