"""
Tests for Sequential model.
"""

import numpy as np
import pytest

from neura_command.layers import Dense, Dropout, ReLU, Softmax, BatchNorm
from neura_command.network.sequential import Sequential
from neura_command.network_utils.optimizers import Adam, SGD
from neura_command.network_utils.loss_functions import MSELoss, CrossEntropyLoss
from neura_command.network_utils.callbacks import EarlyStopping, HistoryCallback


class TestSequentialBasic:
    def test_init_empty(self):
        model = Sequential()
        assert len(model.layers) == 0

    def test_init_with_layers(self):
        model = Sequential([
            Dense(10, 5),
            ReLU(),
            Dense(5, 2),
        ])
        assert len(model.layers) == 3

    def test_add_layer(self):
        model = Sequential()
        model.add(Dense(10, 5))
        model.add(ReLU())
        assert len(model.layers) == 2

    def test_add_chaining(self):
        model = Sequential().add(Dense(10, 5)).add(ReLU()).add(Dense(5, 2))
        assert len(model.layers) == 3


class TestSequentialForward:
    def test_forward_shape(self):
        model = Sequential([
            Dense(10, 5),
            ReLU(),
            Dense(5, 2),
        ])
        x = np.random.randn(32, 10)
        output = model.forward(x)
        assert output.shape == (32, 2)

    def test_predict(self):
        model = Sequential([
            Dense(10, 5),
            Dense(5, 2),
        ])
        x = np.random.randn(32, 10)
        output = model.predict(x)
        assert output.shape == (32, 2)

    def test_predict_uses_eval_mode(self):
        model = Sequential([
            Dense(10, 5),
            Dropout(0.5),
            Dense(5, 2),
        ])
        x = np.ones((100, 10))
        
        # In predict mode, dropout should not drop any values
        output1 = model.predict(x)
        output2 = model.predict(x)
        np.testing.assert_array_equal(output1, output2)


class TestSequentialTraining:
    def test_train_reduces_loss(self):
        np.random.seed(42)
        model = Sequential(
            layers=[
                Dense(4, 8),
                ReLU(),
                Dense(8, 2),
            ],
            optimizer=Adam(learning_rate=0.01),
            loss=MSELoss(),
        )
        
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 2)
        
        initial_loss = model.evaluate(X, y)['loss']
        model.train(X, y, epochs=20, batch_size=16, verbose=False)
        final_loss = model.evaluate(X, y)['loss']
        
        assert final_loss < initial_loss

    def test_train_with_validation(self):
        np.random.seed(42)
        model = Sequential(
            layers=[Dense(4, 2)],
            optimizer=Adam(learning_rate=0.01),
            loss=MSELoss(),
        )
        
        X_train = np.random.randn(80, 4)
        y_train = np.random.randn(80, 2)
        X_val = np.random.randn(20, 4)
        y_val = np.random.randn(20, 2)
        
        history = model.train(
            X_train, y_train,
            epochs=5,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=False,
        )
        
        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) == 5

    def test_train_with_early_stopping(self):
        np.random.seed(42)
        model = Sequential(
            layers=[Dense(4, 2)],
            optimizer=Adam(learning_rate=0.001),  # Slower learning rate
            loss=MSELoss(),
        )
        
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 2)
        
        # Very strict early stopping - will trigger on small improvements
        early_stop = EarlyStopping(patience=3, monitor='loss', min_delta=0.1)
        
        history = model.train(
            X, y,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=False,
        )
        
        # Should stop early due to min_delta threshold
        assert len(history['loss']) < 100

    def test_train_with_sgd(self):
        np.random.seed(42)
        model = Sequential(
            layers=[Dense(4, 2)],
            optimizer=SGD(learning_rate=0.01, momentum=0.9),
            loss=MSELoss(),
        )
        
        X = np.random.randn(50, 4)
        y = np.random.randn(50, 2)
        
        initial_loss = model.evaluate(X, y)['loss']
        model.train(X, y, epochs=10, batch_size=16, verbose=False)
        final_loss = model.evaluate(X, y)['loss']
        
        assert final_loss < initial_loss


class TestSequentialWithDropout:
    def test_dropout_active_during_training(self):
        np.random.seed(42)
        model = Sequential([
            Dense(10, 10),
            Dropout(0.5),
        ])
        
        x = np.ones((100, 10))
        
        # Multiple forward passes during training should give different results
        model.train_mode()
        out1 = model.forward(x, training=True)
        out2 = model.forward(x, training=True)
        
        assert not np.allclose(out1, out2)


class TestSequentialWithBatchNorm:
    def test_batchnorm_in_network(self):
        model = Sequential([
            Dense(10, 20),
            BatchNorm(20),
            ReLU(),
            Dense(20, 5),
        ])
        
        x = np.random.randn(32, 10)
        output = model.forward(x)
        assert output.shape == (32, 5)

    def test_batchnorm_training(self):
        np.random.seed(42)
        model = Sequential(
            layers=[
                Dense(4, 8),
                BatchNorm(8),
                ReLU(),
                Dense(8, 2),
            ],
            optimizer=Adam(learning_rate=0.01),
            loss=MSELoss(),
        )
        
        X = np.random.randn(100, 4)
        y = np.random.randn(100, 2)
        
        initial_loss = model.evaluate(X, y)['loss']
        model.train(X, y, epochs=10, batch_size=32, verbose=False)
        final_loss = model.evaluate(X, y)['loss']
        
        assert final_loss < initial_loss


class TestSequentialSummary:
    def test_summary(self):
        model = Sequential([
            Dense(784, 256),
            ReLU(),
            Dropout(0.2),
            Dense(256, 10),
            Softmax(),
        ])
        
        summary = model.summary()
        assert 'Dense' in summary
        assert 'ReLU' in summary
        assert 'Dropout' in summary
        assert 'Total parameters' in summary

    def test_num_parameters(self):
        model = Sequential([
            Dense(10, 5),  # 10*5 + 5 = 55 params
            Dense(5, 2),   # 5*2 + 2 = 12 params
        ])
        # Total: 67 params
        assert model.num_parameters == 67


class TestSequentialClassification:
    """Test typical classification workflow."""
    
    def test_mnist_like_architecture(self):
        """Test architecture similar to MNIST classifier."""
        model = Sequential(
            layers=[
                Dense(784, 128),
                ReLU(),
                Dropout(0.2),
                Dense(128, 64),
                ReLU(),
                Dropout(0.2),
                Dense(64, 10),
                Softmax(),
            ],
            optimizer=Adam(learning_rate=0.001),
            loss=CrossEntropyLoss(),
        )
        
        # Random "image" data
        X = np.random.randn(32, 784)
        
        # Forward pass should work
        output = model.predict(X)
        assert output.shape == (32, 10)
        
        # Outputs should be valid probabilities
        assert np.all(output >= 0)
        assert np.all(output <= 1)
        np.testing.assert_array_almost_equal(np.sum(output, axis=1), np.ones(32))
