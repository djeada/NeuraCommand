"""
Tests for the new CLI analytics commands.
"""

import json
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

from neura_command.layers import Dense, ReLU
from neura_command.network.sequential import Sequential
from neura_command.network_utils.optimizers import Adam
from neura_command.network_utils.loss_functions import MSELoss


class TestAnalyzeCommand:
    """Tests for the analyze command functionality."""

    @pytest.fixture
    def model_with_data(self, tmp_path):
        """Create a test model and data files."""
        np.random.seed(42)
        
        # Create and train model
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
        
        # Save model
        model_path = tmp_path / "test_model"
        model.save(str(model_path))
        
        # Save test data
        features_path = tmp_path / "features.csv"
        targets_path = tmp_path / "targets.csv"
        np.savetxt(features_path, X[:10], delimiter=",")
        np.savetxt(targets_path, y[:10], delimiter=",")
        
        return {
            "model": model,
            "model_path": str(model_path),
            "features_path": str(features_path),
            "targets_path": str(targets_path),
        }

    def test_analyze_architecture(self, model_with_data):
        """Test architecture analysis functionality."""
        from app.commands.analyze_command import AnalyzeCommand
        
        # Create mock args
        class MockArgs:
            model = model_with_data["model_path"]
            features_csv = None
            targets_csv = None
            analysis_type = "architecture"
            format = "json"
            task_type = "regression"
        
        cmd = AnalyzeCommand(MockArgs())
        # We can't easily capture stdout, but we can test internal methods
        from neura_command.serialization import load_model, get_model_info
        
        model = load_model(model_with_data["model_path"])
        info = get_model_info(model_with_data["model_path"])
        
        arch = cmd._analyze_architecture(model, info)
        
        assert arch["num_layers"] == 3
        assert arch["total_parameters"] == 58  # 4*8+8 + 8*2+2 = 40+18
        assert arch["optimizer"] == "Adam"
        assert arch["loss_function"] == "MSELoss"
        assert len(arch["layers"]) == 3

    def test_analyze_weights(self, model_with_data):
        """Test weight statistics analysis."""
        from app.commands.analyze_command import AnalyzeCommand
        from neura_command.serialization import load_model
        
        class MockArgs:
            model = model_with_data["model_path"]
            features_csv = None
            targets_csv = None
            analysis_type = "weights"
            format = "text"
            task_type = "regression"
        
        cmd = AnalyzeCommand(MockArgs())
        model = load_model(model_with_data["model_path"])
        
        weights = cmd._analyze_weights(model)
        
        assert "global" in weights
        assert "layers" in weights
        assert weights["global"]["total_parameters"] == 58
        assert len(weights["layers"]) == 4  # 2 Dense layers, each with 2 params (weights + bias)

    def test_analyze_performance_regression(self, model_with_data):
        """Test regression performance analysis."""
        from app.commands.analyze_command import AnalyzeCommand
        from neura_command.serialization import load_model
        
        class MockArgs:
            model = model_with_data["model_path"]
            features_csv = model_with_data["features_path"]
            targets_csv = model_with_data["targets_path"]
            analysis_type = "performance"
            format = "text"
            task_type = "regression"
        
        cmd = AnalyzeCommand(MockArgs())
        model = load_model(model_with_data["model_path"])
        features = np.loadtxt(model_with_data["features_path"], delimiter=",")
        targets = np.loadtxt(model_with_data["targets_path"], delimiter=",")
        
        perf = cmd._analyze_performance(model, features, targets, "regression")
        
        assert perf["num_samples"] == 10
        assert "mse" in perf
        assert "rmse" in perf
        assert "mae" in perf
        assert "r2_score" in perf

    def test_analyze_errors(self, model_with_data):
        """Test error analysis."""
        from app.commands.analyze_command import AnalyzeCommand
        from neura_command.serialization import load_model
        
        class MockArgs:
            pass
        
        cmd = AnalyzeCommand(MockArgs())
        model = load_model(model_with_data["model_path"])
        features = np.loadtxt(model_with_data["features_path"], delimiter=",")
        targets = np.loadtxt(model_with_data["targets_path"], delimiter=",")
        
        errors = cmd._analyze_errors(model, features, targets, "regression")
        
        assert errors["num_samples"] == 10
        assert "error_distribution" in errors
        assert "error_percentiles" in errors
        assert "worst_predictions" in errors
        assert len(errors["worst_predictions"]) == 5

    def test_regression_metrics_calculation(self):
        """Test regression metrics calculations."""
        from app.commands.analyze_command import AnalyzeCommand
        
        class MockArgs:
            pass
        
        cmd = AnalyzeCommand(MockArgs())
        
        # Perfect predictions
        y_true = np.array([[1.0], [2.0], [3.0]])
        y_pred = np.array([[1.0], [2.0], [3.0]])
        
        metrics = cmd._regression_metrics(y_true, y_pred)
        
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2_score"] == 1.0

    def test_classification_metrics_calculation(self):
        """Test classification metrics calculations."""
        from app.commands.analyze_command import AnalyzeCommand
        
        class MockArgs:
            pass
        
        cmd = AnalyzeCommand(MockArgs())
        
        # Some predictions (one-hot encoded)
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]])  # All correct
        
        metrics = cmd._classification_metrics(y_true, y_pred)
        
        assert metrics["accuracy"] == 1.0
        assert metrics["num_classes"] == 2

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        from app.commands.analyze_command import AnalyzeCommand
        
        class MockArgs:
            pass
        
        cmd = AnalyzeCommand(MockArgs())
        
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        classes = np.array([0, 1])
        
        matrix = cmd._confusion_matrix(y_true, y_pred, classes)
        
        assert matrix[0, 0] == 1  # TN
        assert matrix[0, 1] == 1  # FP
        assert matrix[1, 0] == 1  # FN
        assert matrix[1, 1] == 1  # TP


class TestVisualizeCommand:
    """Tests for the visualize command functionality."""

    @pytest.fixture
    def history_file(self, tmp_path):
        """Create a test history file."""
        history = {
            "loss": [1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25],
            "val_loss": [1.1, 0.9, 0.7, 0.6, 0.55, 0.5, 0.48, 0.47, 0.46, 0.45],
            "epoch": list(range(10)),
        }
        
        history_path = tmp_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f)
        
        return str(history_path)

    def test_visualize_loads_history(self, history_file):
        """Test that visualize command loads history correctly."""
        with open(history_file, "r") as f:
            history = json.load(f)
        
        assert "loss" in history
        assert "val_loss" in history
        assert len(history["loss"]) == 10

    def test_output_json_format(self, history_file, capsys):
        """Test JSON output format."""
        from app.commands.visualize_command import VisualizeCommand
        
        class MockArgs:
            history_json = history_file
            metric = "loss"
            width = 60
            height = 20
            format = "json"
        
        cmd = VisualizeCommand(MockArgs())
        cmd.execute()
        
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        
        assert "loss" in output
        assert output["loss"]["epochs"] == 10
        assert output["loss"]["initial"] == 1.0
        assert output["loss"]["final"] == 0.25

    def test_output_table_format(self, history_file, capsys):
        """Test table output format."""
        from app.commands.visualize_command import VisualizeCommand
        
        class MockArgs:
            history_json = history_file
            metric = "loss"
            width = 60
            height = 20
            format = "table"
        
        cmd = VisualizeCommand(MockArgs())
        cmd.execute()
        
        captured = capsys.readouterr()
        
        assert "Epoch" in captured.out
        assert "loss" in captured.out
        assert "val_loss" in captured.out

    def test_output_ascii_format(self, history_file, capsys):
        """Test ASCII chart output."""
        from app.commands.visualize_command import VisualizeCommand
        
        class MockArgs:
            history_json = history_file
            metric = "loss"
            width = 40
            height = 10
            format = "ascii"
        
        cmd = VisualizeCommand(MockArgs())
        cmd.execute()
        
        captured = capsys.readouterr()
        
        assert "Training Progress Visualization" in captured.out
        assert "loss" in captured.out
        assert "Initial:" in captured.out
        assert "Final:" in captured.out

    def test_save_training_history_util(self, tmp_path):
        """Test the save_training_history utility function."""
        from app.commands.visualize_command import save_training_history
        
        history = {
            "loss": [1.0, 0.5, 0.3],
            "val_loss": [1.1, 0.6, 0.4],
        }
        
        path = str(tmp_path / "test_history.json")
        save_training_history(history, path)
        
        with open(path, "r") as f:
            loaded = json.load(f)
        
        assert loaded == history
