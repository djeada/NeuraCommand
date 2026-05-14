"""
Tests for legacy and compatibility paths.
"""

import importlib
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from app.commands.create_network_command import CreateNetworkCommand
from app.commands.predict_command import PredictCommand
from app.commands.train_network_command import TrainNetworkCommand
from app.menu.architecture_management_menu.io_menu.load_json_menu import LoadJsonMenu
from app.menu.architecture_management_menu.io_menu.save_json_menu import SaveNetworkMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class DummyPredictModel:
    def predict(self, batch):
        return np.sum(batch, axis=1)


class TestLegacyBuilderCompatibility:
    def test_from_json_accepts_dict_and_builds_loss_instance(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder = NetworkBuilder.from_json({"layers": [], "loss_func": "difference"})
            model = builder.build()

        assert model.loss_func.__class__.__name__ == "DifferenceLoss"

    def test_deprecated_import_is_quiet_until_used(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import neura_command.deprecated as deprecated_module

            importlib.reload(deprecated_module)

        assert caught == []


class TestLegacyJsonMenus:
    def test_save_and_load_json_menu_roundtrip(self, tmp_path):
        with pytest.deprecated_call(match="legacy code"):
            builder = NetworkBuilder(
                {
                    "layers": [
                        {
                            "perceptrons": [
                                {
                                    "input_size": 2,
                                    "output_size": 1,
                                    "activation_func": "relu",
                                    "loss_func": "mse",
                                    "learning_rate": 0.1,
                                }
                            ]
                        }
                    ],
                    "loss_func": "mse",
                }
            )

        save_menu = SaveNetworkMenu(parent_menu=None, network_builder=builder)
        save_menu.file_path = tmp_path / "network.json"
        save_menu.save_network_configuration()

        payload = json.loads(save_menu.file_path.read_text(encoding="utf-8"))
        assert payload == builder.config

        load_menu = LoadJsonMenu(parent_menu=None)
        load_menu.file_path = save_menu.file_path
        load_menu.load_json()

        assert load_menu.result.config == builder.config


class TestCompatibilityCommands:
    def test_create_command_supports_dense_activation_and_difference_loss(self):
        args = SimpleNamespace(quiet=True, verbose=False)
        command = CreateNetworkCommand(args)

        dense = command._create_layer(
            {
                "type": "dense",
                "in_features": 4,
                "out_features": 2,
                "activation": "relu",
            }
        )
        loss = command._create_loss({"class": "DifferenceLoss"})

        assert dense.activation.__class__.__name__ == "ReLUActivation"
        assert loss.__class__.__name__ == "DifferenceLoss"

    def test_train_command_splits_validation_without_leakage(self):
        args = SimpleNamespace(
            validation_split=0.25,
            output="trained",
            overwrite=False,
            batch_size=2,
            quiet=True,
            verbose=False,
        )
        command = TrainNetworkCommand(args)

        features = np.arange(16).reshape(8, 2)
        targets = np.arange(8).reshape(8, 1)

        train_X, train_y, val_X, val_y = command._split_validation_data(features, targets)

        assert len(train_X) == 6
        assert len(val_X) == 2
        assert {tuple(row) for row in train_X}.isdisjoint({tuple(row) for row in val_X})
        assert {tuple(row) for row in train_y}.isdisjoint({tuple(row) for row in val_y})

    def test_train_command_requires_output_destination_before_loading(self):
        args = SimpleNamespace(
            quiet=True,
            verbose=False,
            output=None,
            overwrite=False,
        )
        command = TrainNetworkCommand(args)

        with patch.object(command, "_load_model") as load_model:
            with pytest.raises(SystemExit) as exc_info:
                command.execute()

        assert exc_info.value.code == 1
        load_model.assert_not_called()

    def test_train_command_passes_validation_data_to_modern_models(self):
        args = SimpleNamespace(
            epochs=3,
            batch_size=2,
            verbose=False,
            validation_split=0.25,
            output="trained",
            overwrite=False,
            quiet=True,
            early_stopping=None,
        )
        command = TrainNetworkCommand(args)

        class DummyModel:
            def train(self, features, targets, **kwargs):
                self.features = features
                self.targets = targets
                self.kwargs = kwargs
                return {"loss": [1.0], "val_loss": [2.0]}

        model = DummyModel()
        features = np.arange(16).reshape(8, 2)
        targets = np.arange(8).reshape(8, 1)
        train_X, train_y, val_X, val_y = command._split_validation_data(features, targets)

        history = command._train_model(
            model,
            train_X,
            train_y,
            val_X,
            val_y,
            callbacks=None,
        )

        assert "validation_data" in model.kwargs
        assert len(model.kwargs["validation_data"][0]) == 2
        assert len(model.features) == 6
        assert history["val_loss"] == [2.0]

    def test_predict_command_batches_single_output_models_consistently(self):
        args = SimpleNamespace(
            batch_size=2,
            verbose=False,
            quiet=True,
        )
        command = PredictCommand(args)
        features = np.arange(10).reshape(5, 2)

        predictions = command._make_predictions(DummyPredictModel(), features)

        assert predictions.shape == (5, 1)
        np.testing.assert_array_equal(predictions[:, 0], np.sum(features, axis=1))
