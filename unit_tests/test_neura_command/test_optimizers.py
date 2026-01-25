import numpy as np
import pytest

from neura_command.network_utils.optimizers import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    AdaGrad,
    Adadelta,
    Nadam,
)


class TestSGD:
    def test_basic_update(self):
        optimizer = SGD(learning_rate=0.1)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        new_weights = optimizer.update(param_id, weights, gradient)
        expected = weights - 0.1 * gradient
        np.testing.assert_array_almost_equal(new_weights, expected)

    def test_momentum(self):
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        # First update
        weights1 = optimizer.update(param_id, weights, gradient)

        # Second update should include momentum
        weights2 = optimizer.update(param_id, weights1, gradient)

        # With momentum, the update should be larger
        assert np.all(np.abs(weights2 - weights1) >= np.abs(weights1 - weights) * 0.9)

    def test_reset(self):
        optimizer = SGD(learning_rate=0.1, momentum=0.9)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer.update(param_id, weights, gradient)
        assert len(optimizer._velocity) > 0

        optimizer.reset()
        assert len(optimizer._velocity) == 0


class TestAdam:
    def test_basic_update(self):
        optimizer = Adam(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        # Manually increment timestep for single update test
        optimizer._t = 1
        new_weights = optimizer.update(param_id, weights, gradient)

        # Weights should change
        assert not np.array_equal(new_weights, weights)

    def test_multiple_updates(self):
        optimizer = Adam(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        for i in range(10):
            optimizer._t = i + 1  # Simulate step() incrementing
            weights = optimizer.update(param_id, weights, gradient)

        assert len(optimizer._m) == 1

    def test_reset(self):
        optimizer = Adam(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer._t = 1
        optimizer.update(param_id, weights, gradient)
        assert len(optimizer._m) > 0
        assert len(optimizer._v) > 0

        optimizer.reset()
        assert len(optimizer._m) == 0
        assert len(optimizer._v) == 0
        assert optimizer._t == 0

    def test_adaptive_learning(self):
        """Test that Adam adapts learning rate per parameter."""
        optimizer = Adam(learning_rate=0.1)
        weights = np.array([1.0, 1.0])
        param_id = id(weights)

        # Different gradients for different parameters
        gradient1 = np.array([1.0, 0.01])  # First param has larger gradient

        optimizer._t = 1
        weights = optimizer.update(param_id, weights, gradient1)

        # Both should be updated, but proportionally different
        assert weights[0] != 1.0
        assert weights[1] != 1.0


class TestRMSprop:
    def test_basic_update(self):
        optimizer = RMSprop(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        new_weights = optimizer.update(param_id, weights, gradient)
        assert not np.array_equal(new_weights, weights)

    def test_reset(self):
        optimizer = RMSprop(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer.update(param_id, weights, gradient)
        assert len(optimizer._avg_sq_grad) > 0

        optimizer.reset()
        assert len(optimizer._avg_sq_grad) == 0


class TestAdaGrad:
    def test_basic_update(self):
        optimizer = AdaGrad(learning_rate=0.1)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        new_weights = optimizer.update(param_id, weights, gradient)
        assert not np.array_equal(new_weights, weights)

    def test_learning_rate_decay(self):
        """AdaGrad should effectively reduce learning rate over time."""
        optimizer = AdaGrad(learning_rate=0.1)
        weights = np.array([1.0])
        gradient = np.array([1.0])
        param_id = id(weights)

        updates = []
        for _ in range(5):
            new_weights = optimizer.update(param_id, weights, gradient)
            updates.append(abs(new_weights[0] - weights[0]))
            weights = new_weights

        # Updates should decrease over time
        for i in range(1, len(updates)):
            assert updates[i] < updates[i - 1]

    def test_reset(self):
        optimizer = AdaGrad(learning_rate=0.1)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer.update(param_id, weights, gradient)
        assert len(optimizer._accumulated_grad) > 0

        optimizer.reset()
        assert len(optimizer._accumulated_grad) == 0


class TestAdadelta:
    def test_basic_update(self):
        optimizer = Adadelta(rho=0.95)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        new_weights = optimizer.update(param_id, weights, gradient)
        assert not np.array_equal(new_weights, weights)

    def test_no_learning_rate_dependency(self):
        """Adadelta should work without explicit learning rate tuning."""
        optimizer = Adadelta(rho=0.95)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        # Multiple updates should continue to make progress
        for _ in range(10):
            new_weights = optimizer.update(param_id, weights, gradient)
            weights = new_weights

        # Should have moved from initial position
        assert not np.allclose(weights, [1.0, 2.0, 3.0])

    def test_reset(self):
        optimizer = Adadelta(rho=0.95)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer.update(param_id, weights, gradient)
        assert len(optimizer._avg_sq_grad) > 0
        assert len(optimizer._avg_sq_delta) > 0

        optimizer.reset()
        assert len(optimizer._avg_sq_grad) == 0
        assert len(optimizer._avg_sq_delta) == 0


class TestNadam:
    def test_basic_update(self):
        optimizer = Nadam(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer._t = 1
        new_weights = optimizer.update(param_id, weights, gradient)
        assert not np.array_equal(new_weights, weights)

    def test_multiple_updates(self):
        optimizer = Nadam(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        for i in range(10):
            optimizer._t = i + 1
            weights = optimizer.update(param_id, weights, gradient)

        assert len(optimizer._m) == 1
        assert len(optimizer._v) == 1

    def test_reset(self):
        optimizer = Nadam(learning_rate=0.001)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.1, 0.2, 0.3])
        param_id = id(weights)

        optimizer._t = 1
        optimizer.update(param_id, weights, gradient)
        assert len(optimizer._m) > 0
        assert len(optimizer._v) > 0

        optimizer.reset()
        assert len(optimizer._m) == 0
        assert len(optimizer._v) == 0
        assert optimizer._t == 0


class TestSGDL1Regularization:
    def test_l1_regularization(self):
        """Test that L1 regularization pushes weights toward zero."""
        optimizer = SGD(learning_rate=0.1, l1_regularization=0.1)
        weights = np.array([1.0, -1.0, 0.5])
        gradient = np.array([0.0, 0.0, 0.0])  # Zero gradient
        param_id = id(weights)

        new_weights = optimizer.update(param_id, weights, gradient)

        # Positive weights should decrease, negative should increase (toward zero)
        assert new_weights[0] < weights[0]
        assert new_weights[1] > weights[1]
        assert new_weights[2] < weights[2]

    def test_l1_l2_combined(self):
        """Test combined L1 and L2 regularization."""
        optimizer = SGD(learning_rate=0.1, weight_decay=0.1, l1_regularization=0.1)
        weights = np.array([1.0, 2.0, 3.0])
        gradient = np.array([0.0, 0.0, 0.0])
        param_id = id(weights)

        new_weights = optimizer.update(param_id, weights, gradient)

        # All weights should decrease due to regularization
        assert np.all(new_weights < weights)
