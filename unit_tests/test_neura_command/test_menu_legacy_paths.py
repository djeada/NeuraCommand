"""
Tests for remaining legacy menu compatibility paths.
"""

import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from app.menu.architecture_management_menu.architecture_management_menu import (
    ArchitectureManagementMenu,
)
from app.menu.model_management_menu.io_menu.load_pickle_menu import LoadPickleMenu
from app.menu.model_management_menu.io_menu.save_pickle_menu import SavePickleMenu
from app.menu.model_management_menu.use_model_menu.make_prediction_menu import (
    MakePredictionMenu,
)
from app.menu.model_management_menu.use_model_menu.train_network_menu import (
    TrainNetworkMenu,
)


class DummyLegacyPredictor:
    input_size = 3
    output_size = 1

    def predict(self, data):
        assert data.ndim == 2
        return np.sum(data, axis=1, keepdims=True)

    def train(self, X, y, epochs, batch_size):
        self.last_train = (X, y, epochs, batch_size)


class TestLegacyMenuPaths:
    def test_architecture_menu_init_is_quiet_until_legacy_flow_is_used(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            menu = ArchitectureManagementMenu(parent_menu=None)

        assert menu.builder is None
        assert caught == []

    def test_load_pickle_menu_accepts_pkl_default(self, tmp_path):
        pickle_path = tmp_path / "network.pkl"
        pickle_path.write_bytes(b"test")
        menu = LoadPickleMenu(parent_menu=None)

        with patch("builtins.input", return_value=str(pickle_path.with_suffix(""))):
            menu.prompt_for_path()

        assert menu.file_path == pickle_path
        assert menu.is_active is True

    def test_save_pickle_menu_appends_default_extension_and_creates_parent(self, tmp_path):
        menu = SavePickleMenu(parent_menu=None, network={"weights": [1, 2, 3]})

        with patch("builtins.input", return_value=str(tmp_path / "artifacts" / "model")):
            menu.prompt_for_path()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            menu.save_network_to_pickle()

        assert menu.file_path == tmp_path / "artifacts" / "model.pkl"
        assert menu.file_path.exists()

    def test_make_prediction_menu_reshapes_manual_single_sample(self):
        menu = MakePredictionMenu(parent_menu=None, network=DummyLegacyPredictor())
        menu.method = "manual"

        with patch("builtins.input", return_value="1,2,3"):
            menu.load_or_enter_data()

        menu.make_prediction()

        assert menu.data.shape == (1, 3)
        assert menu.prediction.shape == (1, 1)
        np.testing.assert_array_equal(menu.prediction, [[6.0]])

    def test_make_prediction_menu_reshapes_single_csv_row(self, tmp_path):
        csv_path = tmp_path / "single.csv"
        np.savetxt(csv_path, np.array([1.0, 2.0, 3.0]), delimiter=",")
        menu = MakePredictionMenu(parent_menu=None, network=DummyLegacyPredictor())

        data = menu.load_data_from_csv(str(csv_path))

        assert data.shape == (1, 3)

    def test_train_network_menu_handles_single_csv_row(self, tmp_path):
        csv_path = tmp_path / "train.csv"
        np.savetxt(csv_path, np.array([1.0, 2.0, 3.0, 0.5]), delimiter=",")
        menu = TrainNetworkMenu(parent_menu=None, network=DummyLegacyPredictor())

        with patch("builtins.input", return_value=str(csv_path)):
            X_train, y_train = menu.load_data_from_csv()

        assert X_train.shape == (1, 3)
        assert y_train.shape == (1, 1)

    def test_train_network_menu_empty_manual_input_returns_none(self):
        menu = TrainNetworkMenu(parent_menu=None, network=DummyLegacyPredictor())

        with patch("builtins.input", side_effect=["done"]):
            X_train, y_train = menu.enter_data_manually()

        assert X_train is None
        assert y_train is None

    def test_train_network_menu_retries_invalid_parameters(self):
        menu = TrainNetworkMenu(parent_menu=None, network=DummyLegacyPredictor())

        with patch("builtins.input", side_effect=["0", "2", "5", "2"]):
            menu.set_training_parameters()

        assert menu.epochs == 5
        assert menu.batch_size == 2
