"""
End-to-end CLI workflow tests.
"""

import json
from pathlib import Path

import numpy as np

from neura_command.cli import main as cli_main
from neura_command.inference import InferenceModel
from neura_command.serialization import load_model


class TestCLIWorkflow:
    def test_create_train_predict_export_workflow(self, tmp_path):
        architecture = {
            "layers": [
                {"type": "dense", "in_features": 2, "out_features": 4},
                {"type": "relu"},
                {"type": "dense", "in_features": 4, "out_features": 1},
            ],
            "optimizer": {"class": "Adam", "learning_rate": 0.01},
            "loss": {"class": "MSELoss"},
        }
        architecture_path = tmp_path / "architecture.json"
        architecture_path.write_text(json.dumps(architecture))

        model_path = tmp_path / "model"
        trained_path = tmp_path / "trained_model"
        prediction_path = tmp_path / "predictions.json"
        export_path = tmp_path / "exported_model"
        history_path = tmp_path / "history.json"
        features_path = tmp_path / "features.csv"
        targets_path = tmp_path / "targets.csv"

        X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=np.float32,
        )
        y = (X[:, :1] * 0.75 + X[:, 1:] * 0.25).astype(np.float32)
        np.savetxt(features_path, X, delimiter=",")
        np.savetxt(targets_path, y, delimiter=",")

        assert cli_main(
            [
                "create",
                "--architecture",
                str(architecture_path),
                "--output",
                str(model_path),
                "--quiet",
            ]
        ) == 0
        assert model_path.with_suffix(".json").exists()
        assert model_path.with_suffix(".npz").exists()

        assert cli_main(
            [
                "train",
                "--model",
                str(model_path),
                "--features-csv",
                str(features_path),
                "--targets-csv",
                str(targets_path),
                "--epochs",
                "25",
                "--batch-size",
                "2",
                "--output",
                str(trained_path),
                "--save-history",
                str(history_path),
                "--quiet",
            ]
        ) == 0
        assert trained_path.with_suffix(".json").exists()
        assert trained_path.with_suffix(".npz").exists()
        assert history_path.exists()

        assert cli_main(
            [
                "predict",
                "--model",
                str(trained_path),
                "--input-csv",
                str(features_path),
                "--output-mode",
                "json",
                "--output-json",
                str(prediction_path),
                "--batch-size",
                "2",
                "--quiet",
            ]
        ) == 0
        assert prediction_path.exists()
        prediction_payload = json.loads(prediction_path.read_text())
        assert prediction_payload["num_samples"] == len(X)
        assert len(prediction_payload["predictions"]) == len(X)

        assert cli_main(
            [
                "export",
                "--model",
                str(trained_path),
                "--output",
                str(export_path),
                "--quantize",
                "float16",
            ]
        ) == 0

        inference_model = InferenceModel.load(str(export_path))
        trained_model = load_model(str(trained_path))
        inference_predictions = inference_model.predict(X)
        model_predictions = trained_model.predict(X)

        assert Path(f"{export_path}.json").exists()
        assert Path(f"{export_path}.npz").exists()
        assert inference_predictions.shape == model_predictions.shape
        np.testing.assert_allclose(
            inference_predictions,
            model_predictions,
            atol=1e-2,
            rtol=1e-2,
        )
