"""CLI errors, reproducibility, machine output, and non-overwriting companions."""

import json

import numpy as np
import pytest

from neura_command import Sequential, Dense, Softmax
from neura_command.cli import main as cli_main
from neura_command.inference import InferenceModel
from neura_command.serialization import load_model
from neura_command.serialization.model_io import model_paths


@pytest.fixture
def inputs(tmp_path):
    model = Sequential([Dense(2, 2), Softmax()])
    model.save(tmp_path / "model")
    features, targets = tmp_path / "features.csv", tmp_path / "targets.csv"
    features.write_text("1,2\n3,4\n")
    targets.write_text("1,0\n0,1\n")
    return tmp_path, str(tmp_path / "model"), str(features), str(targets)


@pytest.mark.parametrize(
    "command,flags",
    [
        ("train", ["--epochs", "0"]),
        ("train", ["--batch-size", "-1"]),
        ("train", ["--validation-split", "1"]),
        ("train", ["--validation-split", "nan"]),
        ("train", ["--lr", "inf"]),
        ("train", ["--seed", str(2**32)]),
        ("train", ["--early-stopping", "0"]),
        ("predict", ["--top-k", "0"]),
        ("predict", ["--threshold", "1.2"]),
        ("predict", ["--precision", "17"]),
        ("predict", ["--batch-size", "0"]),
        ("export", ["--max-weight-bytes", "-1"]),
        ("export", ["--prune-threshold", "nan"]),
        ("export", ["--max-weight-byt", "1024"]),
    ],
)
def test_invalid_arguments_fail_during_parsing(inputs, command, flags):
    directory, model, features, targets = inputs
    arguments = [command, "--model", model]
    if command == "train":
        arguments += [
            "--features-csv",
            features,
            "--targets-csv",
            targets,
            "--output",
            str(directory / "out"),
        ]
    elif command == "predict":
        arguments += ["--input-csv", features]
    else:
        arguments += ["--output", str(directory / "out")]
    with pytest.raises(SystemExit) as error:
        cli_main(arguments + flags)
    assert error.value.code == 2
    assert not (directory / "out.npz").exists()


@pytest.mark.parametrize("prefix", [["--action", "info"], ["--action=info"]])
def test_legacy_action_uses_the_same_strict_parser(inputs, prefix, capsys):
    _, model, _, _ = inputs
    assert cli_main([*prefix, "--model", model, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_type"] == "training_model"
    with pytest.raises(SystemExit) as error:
        cli_main([*prefix, "--model", model, "--nonsense"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "architecture",
    [
        [],
        {"layers": [4]},
        {"layers": [{"type": "dense", "in_features": 0, "out_features": 2}]},
        {
            "layers": [
                {"type": "dense", "in_features": 2, "out_features": 3},
                {"type": "dense", "in_features": 4, "out_features": 1},
            ]
        },
        {
            "layers": [
                {"type": "input", "shape": [2, 3]},
                {"type": "dense", "in_features": 6, "out_features": 1},
            ]
        },
        {"layers": [{"type": "activation", "activation": 5}]},
        {
            "layers": [
                {
                    "type": "dense",
                    "in_features": 2,
                    "out_features": 1,
                    "activation": "typo",
                }
            ]
        },
        {"layers": [{"type": "batchnorm", "num_features": 2, "epsilon": 0}]},
    ],
)
def test_validate_checks_structure_and_connections_without_creating_files(
    tmp_path, architecture
):
    source = tmp_path / "architecture.json"
    source.write_text(json.dumps(architecture))
    with pytest.raises(SystemExit) as error:
        cli_main(
            [
                "create",
                "--architecture",
                str(source),
                "--validate",
                "--output",
                str(tmp_path / "out"),
                "--quiet",
            ]
        )
    assert error.value.code == 1
    assert not (tmp_path / "out.npz").exists()


@pytest.mark.parametrize("activation,expected", [("linear", -2), ("leakyrelu", -0.6)])
def test_generic_activation_aliases_do_not_fall_back_to_relu(
    tmp_path, activation, expected
):
    source = tmp_path / "architecture.json"
    source.write_text(
        json.dumps(
            {
                "layers": [
                    {"type": "dense", "in_features": 1, "out_features": 1},
                    {"type": "activation", "activation": activation, "alpha": 0.3},
                ]
            }
        )
    )
    assert (
        cli_main(
            [
                "create",
                "--architecture",
                str(source),
                "--output",
                str(tmp_path / "created"),
                "--quiet",
            ]
        )
        == 0
    )
    model = load_model(tmp_path / "created")
    model.layers[0].weights[:] = 1
    model.layers[0].bias[:] = 0
    np.testing.assert_allclose(model.predict([[-2]]), [[expected]], atol=1e-6)


def test_info_and_export_json_support_training_and_inference(inputs, capsys):
    directory, model, _, _ = inputs
    assert cli_main(["info", "--model", model, "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["total_parameters"] == 6
    for format_ in ("inference", "c"):
        output = directory / format_
        assert (
            cli_main(
                [
                    "export",
                    "--model",
                    model,
                    "--output",
                    str(output),
                    "--format",
                    format_,
                    "--quantize",
                    "int8",
                    "--per-channel",
                    "--json",
                ]
            )
            == 0
        )
        report = json.loads(capsys.readouterr().out)
        assert (
            report["exported_parameters"] == 4
        )  # Zero bias removed by exact optimization.
        assert report["weight_bytes"] > 0
        assert report["format"] == format_
    assert (
        cli_main(["info", "--model", str(directory / "inference"), "--format", "json"])
        == 0
    )
    info = json.loads(capsys.readouterr().out)
    assert info["artifact_type"] == "inference_model"
    assert info["total_parameters"] == 4
    assert (
        info["parameter_bytes"]
        == InferenceModel.load(directory / "inference").memory_bytes
    )


def test_json_export_failure_keeps_stdout_empty(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli_main(
            [
                "export",
                "--model",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "out"),
                "--json",
            ]
        )
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error" in captured.err


def test_csv_companion_paths_do_not_overwrite_extensionless_predictions(inputs):
    directory, model, features, _ = inputs
    output = directory / "nested" / "predictions"
    assert (
        cli_main(
            [
                "predict",
                "--model",
                model,
                "--input-csv",
                features,
                "--output-mode",
                "save",
                "--output-csv",
                str(output),
                "--show-confidence",
                "--quiet",
            ]
        )
        == 0
    )
    expected = load_model(model).predict(np.loadtxt(features, delimiter=","))
    np.testing.assert_allclose(np.loadtxt(output, delimiter=","), expected, atol=1e-6)
    assert output.with_name("predictions_classes.csv").exists()
    assert output.with_name("predictions_confidence.csv").exists()


def test_prediction_write_errors_and_missing_destinations_fail(inputs):
    directory, model, features, _ = inputs
    for output_flags in ([], ["--output-csv", str(directory)]):
        with pytest.raises(SystemExit) as error:
            cli_main(
                [
                    "predict",
                    "--model",
                    model,
                    "--input-csv",
                    features,
                    "--output-mode",
                    "save",
                    *output_flags,
                    "--quiet",
                ]
            )
        assert error.value.code == 1


def test_dotted_paths_roundtrip_and_old_path_fallback(tmp_path):
    model = Sequential([Dense(2, 1)])
    path = tmp_path / "model.v2"
    model.save(path)
    assert (tmp_path / "model.v2.json").exists()
    assert (tmp_path / "model.v2.npz").exists()
    expected = model.predict([[1, 2]])
    for reference in (path, str(path) + ".json", str(path) + ".npz"):
        np.testing.assert_array_equal(load_model(reference).predict([[1, 2]]), expected)
    # Models produced before dotted base-name preservation remain readable.
    model.save(tmp_path / "old")
    np.testing.assert_array_equal(
        load_model(tmp_path / "old.version").predict([[1, 2]]), expected
    )
    assert model_paths(path)[0].name == "model.v2.json"


def test_seeded_cli_creation_and_training_are_reproducible(inputs):
    directory, _, features, targets = inputs
    source = directory / "architecture.json"
    source.write_text(
        json.dumps(
            {
                "layers": [{"type": "dense", "in_features": 2, "out_features": 2}],
                "optimizer": "SGD",
            }
        )
    )
    results = []
    for run in range(2):
        base = directory / f"model_{run}"
        trained = directory / f"trained_{run}"
        assert (
            cli_main(
                [
                    "create",
                    "--architecture",
                    str(source),
                    "--output",
                    str(base),
                    "--seed",
                    "12",
                    "--quiet",
                ]
            )
            == 0
        )
        # Batch size larger than the dataset is a valid single partial batch.
        assert (
            cli_main(
                [
                    "train",
                    "--model",
                    str(base),
                    "--features-csv",
                    features,
                    "--targets-csv",
                    targets,
                    "--output",
                    str(trained),
                    "--epochs",
                    "2",
                    "--seed",
                    "7",
                    "--quiet",
                ]
            )
            == 0
        )
        results.append(load_model(trained).get_weights())
    for actual, expected in zip(*results):
        np.testing.assert_array_equal(actual, expected)


def test_info_missing_model_fails_on_stderr(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        cli_main(["info", "--model", str(tmp_path / "missing"), "--format", "json"])
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error" in captured.err


def test_no_shuffle_reserves_the_last_samples_for_validation(tmp_path):
    from neura_command import SGD

    model = Sequential([Dense(1, 1, use_bias=False)], optimizer=SGD(learning_rate=0))
    model.layers[0].weights[:] = 0
    model.save(tmp_path / "initial")
    features, targets = tmp_path / "features.csv", tmp_path / "targets.csv"
    features.write_text("0\n0\n0\n0\n0\n")
    targets.write_text("1\n2\n3\n4\n5\n")
    history_path = tmp_path / "history.json"
    assert (
        cli_main(
            [
                "train",
                "--model",
                str(tmp_path / "initial"),
                "--features-csv",
                str(features),
                "--targets-csv",
                str(targets),
                "--output",
                str(tmp_path / "trained"),
                "--epochs",
                "1",
                "--validation-split",
                "0.4",
                "--no-shuffle",
                "--save-history",
                str(history_path),
                "--quiet",
            ]
        )
        == 0
    )
    history = json.loads(history_path.read_text())
    assert np.isclose(history["loss"][0], 14 / 3)
    assert np.isclose(history["val_loss"][0], 20.5)
