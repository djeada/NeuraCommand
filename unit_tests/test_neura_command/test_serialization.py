"""
Tests for model serialization.
"""

import os
import tempfile
import numpy as np
import pytest

from neura_command import (
    Sequential,
    Dense,
    Dropout,
    BatchNorm,
    ReLU,
    Sigmoid,
    Softmax,
    Adam,
    SGD,
    MSELoss,
    CrossEntropyLoss,
)
from neura_command.serialization import save_model, load_model
from neura_command.serialization.model_io import get_model_info


class TestSaveLoad:
    def test_save_creates_files(self):
        model = Sequential([Dense(10, 5), ReLU()])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)

            assert os.path.exists(path + ".json")
            assert os.path.exists(path + ".npz")

    def test_save_load_roundtrip(self):
        model = Sequential(
            layers=[Dense(10, 5), ReLU(), Dense(5, 2)],
            optimizer=Adam(learning_rate=0.001),
            loss=MSELoss(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert len(loaded.layers) == len(model.layers)
            assert loaded.optimizer.__class__.__name__ == "Adam"
            assert loaded.loss.__class__.__name__ == "MSELoss"

    def test_weights_preserved(self):
        np.random.seed(42)
        model = Sequential([Dense(10, 5), Dense(5, 2)])

        # Get original weights
        original_weights = [
            p.copy() for layer in model.layers for p in layer.parameters()
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            loaded_weights = [p for layer in loaded.layers for p in layer.parameters()]

            for orig, load in zip(original_weights, loaded_weights):
                np.testing.assert_array_equal(orig, load)

    def test_predictions_match(self):
        np.random.seed(42)
        model = Sequential(
            [
                Dense(10, 5),
                ReLU(),
                Dense(5, 2),
                Sigmoid(),
            ]
        )

        X = np.random.randn(20, 10)
        original_pred = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            loaded_pred = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestLayerSerialization:
    def test_dense_layer(self):
        model = Sequential([Dense(10, 5, use_bias=True)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.layers[0].in_features == 10
            assert loaded.layers[0].out_features == 5

    def test_dropout_layer(self):
        model = Sequential([Dense(10, 5), Dropout(0.3)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.layers[1].rate == 0.3

    def test_batchnorm_layer(self):
        model = Sequential([Dense(10, 5), BatchNorm(5)])

        # Do a forward pass to update running stats
        X = np.random.randn(32, 10)
        model.forward(X, training=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            # Check running stats are preserved
            np.testing.assert_array_almost_equal(
                model.layers[1].running_mean, loaded.layers[1].running_mean
            )
            np.testing.assert_array_almost_equal(
                model.layers[1].running_var, loaded.layers[1].running_var
            )

    def test_activation_layers(self):
        model = Sequential(
            [
                Dense(10, 5),
                ReLU(),
                Sigmoid(),
                Softmax(),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.layers[1].__class__.__name__ == "ReLU"
            assert loaded.layers[2].__class__.__name__ == "Sigmoid"
            assert loaded.layers[3].__class__.__name__ == "Softmax"


class TestOptimizerSerialization:
    def test_adam_config(self):
        model = Sequential(
            layers=[Dense(10, 5)],
            optimizer=Adam(learning_rate=0.01, beta1=0.8, beta2=0.99),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.optimizer.learning_rate == 0.01
            assert loaded.optimizer.beta1 == 0.8
            assert loaded.optimizer.beta2 == 0.99

    def test_sgd_config(self):
        model = Sequential(
            layers=[Dense(10, 5)],
            optimizer=SGD(learning_rate=0.1, momentum=0.9),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.optimizer.learning_rate == 0.1
            assert loaded.optimizer.momentum == 0.9


class TestLossSerialization:
    def test_mse_loss(self):
        model = Sequential([Dense(10, 5)], loss=MSELoss())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.loss.__class__.__name__ == "MSELoss"

    def test_crossentropy_loss(self):
        model = Sequential([Dense(10, 5)], loss=CrossEntropyLoss())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)
            loaded = load_model(path)

            assert loaded.loss.__class__.__name__ == "CrossEntropyLoss"


class TestModelInfo:
    def test_get_model_info(self):
        model = Sequential(
            [
                Dense(784, 128),
                ReLU(),
                Dense(128, 10),
            ],
            optimizer=Adam(),
            loss=CrossEntropyLoss(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(model, path)

            info = get_model_info(path)

            assert info["num_layers"] == 3
            assert info["layers"] == ["Dense", "ReLU", "Dense"]
            assert info["optimizer"] == "Adam"
            assert info["loss"] == "CrossEntropyLoss"
            assert info["weights_size_bytes"] > 0


class TestSequentialSaveLoad:
    def test_model_save_method(self):
        model = Sequential([Dense(10, 5)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            model.save(path)

            assert os.path.exists(path + ".json")
            assert os.path.exists(path + ".npz")

    def test_model_load_method(self):
        model = Sequential([Dense(10, 5), ReLU()])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            model.save(path)

            loaded = Sequential.load(path)
            assert len(loaded.layers) == 2


class TestMetadata:
    def test_custom_metadata(self):
        model = Sequential([Dense(10, 5)])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model")
            save_model(
                model,
                path,
                metadata={
                    "description": "Test model",
                    "accuracy": 0.95,
                },
            )

            info = get_model_info(path)
            assert info["metadata"]["description"] == "Test model"
            assert info["metadata"]["accuracy"] == 0.95


class TestTrainedModelSaveLoad:
    def test_trained_model_roundtrip(self):
        """Test that a trained model produces same predictions after save/load."""
        np.random.seed(42)

        # Create and train a model
        model = Sequential(
            layers=[
                Dense(4, 8),
                ReLU(),
                Dense(8, 2),
            ],
            optimizer=Adam(learning_rate=0.01),
            loss=MSELoss(),
        )

        X = np.random.randn(50, 4)
        y = np.random.randn(50, 2)

        model.train(X, y, epochs=5, batch_size=16, verbose=False)

        # Get predictions before save
        pred_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "trained_model")
            model.save(path)
            loaded = Sequential.load(path)

            # Predictions should match exactly
            pred_after = loaded.predict(X)
            np.testing.assert_array_almost_equal(pred_before, pred_after)
