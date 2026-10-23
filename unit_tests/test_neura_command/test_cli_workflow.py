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

        assert (
            cli_main(
                [
                    "create",
                    "--architecture",
                    str(architecture_path),
                    "--output",
                    str(model_path),
                    "--quiet",
                ]
            )
            == 0
        )
        assert model_path.with_suffix(".json").exists()
        assert model_path.with_suffix(".npz").exists()

        assert (
            cli_main(
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
            )
            == 0
        )
        assert trained_path.with_suffix(".json").exists()
        assert trained_path.with_suffix(".npz").exists()
        assert history_path.exists()

        assert (
            cli_main(
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
            )
            == 0
        )
        assert prediction_path.exists()
        prediction_payload = json.loads(prediction_path.read_text())
        assert prediction_payload["num_samples"] == len(X)
        assert len(prediction_payload["predictions"]) == len(X)

        assert (
            cli_main(
                [
                    "export",
                    "--model",
                    str(trained_path),
                    "--output",
                    str(export_path),
                    "--quantize",
                    "float16",
                ]
            )
            == 0
        )

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


class TestEmbeddedCLI:
    def test_device_training_export_predict_and_c_budget_workflow(self, tmp_path):
        import pytest
        from neura_command import Sequential, Dense, ReLU

        model = Sequential([Dense(2, 5), ReLU(), Dense(5, 1)])
        model.save(tmp_path / "initial")
        features = tmp_path / "features.csv"
        targets = tmp_path / "targets.csv"
        np.savetxt(features, [[0, 1], [1, 0], [1, 1], [2, 1]], delimiter=",")
        np.savetxt(targets, [[1], [1], [2], [3]], delimiter=",")
        assert (
            cli_main(
                [
                    "train",
                    "--model",
                    str(tmp_path / "initial"),
                    "--device",
                    "cpu",
                    "--features-csv",
                    str(features),
                    "--targets-csv",
                    str(targets),
                    "--epochs",
                    "3",
                    "--batch-size",
                    "2",
                    "--output",
                    str(tmp_path / "trained"),
                    "--quiet",
                ]
            )
            == 0
        )
        assert (
            cli_main(
                [
                    "export",
                    "--model",
                    str(tmp_path / "trained"),
                    "--output",
                    str(tmp_path / "artifact"),
                    "--quantize",
                    "int8",
                    "--per-channel",
                ]
            )
            == 0
        )
        single = tmp_path / "single.csv"
        single.write_text("0.5,1.5\n")
        assert (
            cli_main(
                [
                    "predict",
                    "--model",
                    str(tmp_path / "artifact"),
                    "--input-csv",
                    str(single),
                    "--device",
                    "cpu",
                    "--output-mode",
                    "json",
                    "--output-json",
                    str(tmp_path / "prediction.json"),
                    "--quiet",
                ]
            )
            == 0
        )
        payload = json.loads((tmp_path / "prediction.json").read_text())
        assert payload["num_samples"] == 1
        assert (
            cli_main(
                [
                    "export",
                    "--model",
                    str(tmp_path / "trained"),
                    "--output",
                    str(tmp_path / "firmware"),
                    "--format",
                    "c",
                    "--quantize",
                    "int8",
                    "--per-channel",
                    "--max-weight-bytes",
                    "256",
                    "--max-scratch-bytes",
                    "128",
                ]
            )
            == 0
        )
        assert (tmp_path / "firmware.c").exists()
        assert (tmp_path / "firmware.h").exists()
        with pytest.raises(SystemExit) as error:
            cli_main(
                [
                    "export",
                    "--model",
                    str(tmp_path / "trained"),
                    "--output",
                    str(tmp_path / "rejected"),
                    "--format",
                    "c",
                    "--quantize",
                    "int8",
                    "--max-weight-bytes",
                    "1",
                ]
            )
        assert error.value.code == 1
        assert not (tmp_path / "rejected.c").exists()

    def test_export_failure_and_unknown_flags_have_nonzero_exit(self, tmp_path):
        import pytest

        for args in (
            [
                "export",
                "--model",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "out"),
            ],
            [
                "export",
                "--model",
                "unused",
                "--output",
                "unused",
                "--invalid-budget",
                "1",
            ],
        ):
            with pytest.raises(SystemExit) as error:
                cli_main(args)
            assert error.value.code != 0

    def test_create_preserves_supported_optimizer_and_activation_config(self, tmp_path):
        architecture = {
            "layers": [
                {"type": "input", "shape": [2]},
                {"type": "flatten"},
                {
                    "type": "dense",
                    "in_features": 2,
                    "out_features": 1,
                    "activation": "leaky_relu",
                    "alpha": 0.3,
                },
            ],
            "optimizer": {"class": "Nadam", "weight_decay": 0.02},
        }
        source = tmp_path / "architecture.json"
        source.write_text(json.dumps(architecture))
        assert (
            cli_main(
                [
                    "create",
                    "--architecture",
                    str(source),
                    "--device",
                    "cpu",
                    "--output",
                    str(tmp_path / "created"),
                    "--quiet",
                ]
            )
            == 0
        )
        loaded = load_model(tmp_path / "created")
        assert loaded.optimizer.__class__.__name__ == "Nadam"
        assert loaded.optimizer.weight_decay == 0.02
        assert loaded.layers[-1].activation.alpha == 0.3

    def test_training_failure_does_not_save_a_success_artifact(self, tmp_path):
        import pytest
        from neura_command import Sequential, Dense

        Sequential([Dense(2, 1)]).save(tmp_path / "initial")
        features, targets = tmp_path / "features.csv", tmp_path / "targets.csv"
        features.write_text("1,2\n3,4\n")
        targets.write_text("1\n2\n")
        with pytest.raises(SystemExit) as error:
            cli_main(
                [
                    "train",
                    "--model",
                    str(tmp_path / "initial"),
                    "--features-csv",
                    str(features),
                    "--targets-csv",
                    str(targets),
                    "--epochs",
                    "0",
                    "--output",
                    str(tmp_path / "failed"),
                    "--quiet",
                ]
            )
        assert error.value.code == 2
        assert not (tmp_path / "failed.npz").exists()
