"""Behavioral checks for export correctness, compaction, and compiled C execution."""

import ctypes
import json
import shutil
import subprocess

import numpy as np
import pytest

from neura_command import (
    Sequential,
    Dense,
    BatchNorm,
    Dropout,
    Flatten,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
    Activation,
)
from neura_command.layers import Input
from neura_command.inference import InferenceModel, QuantizationConfig
from neura_command.inference.quantization import (
    _compute_int8_params,
    _quantize_to_int8,
    _dequantize_from_int8,
)
from neura_command.network_utils.activation_functions import (
    ReLUActivation,
    SigmoidActivation,
    TanhActivation,
    SoftmaxActivation,
    LeakyReLUActivation,
    ELUActivation,
)


@pytest.mark.parametrize(
    "activation",
    [
        ReLUActivation(),
        SigmoidActivation(),
        TanhActivation(),
        SoftmaxActivation(),
        LeakyReLUActivation(0.2),
        ELUActivation(1.3),
    ],
)
@pytest.mark.parametrize("precision", [None, "float16", "int8"])
def test_dense_activation_survives_export_and_reload(tmp_path, activation, precision):
    rng = np.random.default_rng(7)
    model = Sequential([Dense(4, 3, activation=activation)])
    model.layers[0].weights[:] = rng.normal(size=(4, 3)) * 0.2
    x = rng.normal(size=(8, 4)).astype(np.float32)
    expected = model.predict(x)
    exported = InferenceModel.from_sequential(model, quantize=precision)
    exported.save(tmp_path / "artifact")
    assert (
        json.loads((tmp_path / "artifact.json").read_text())["format_version"] == "2.0"
    )
    loaded = InferenceModel.load(tmp_path / "artifact")
    np.testing.assert_allclose(loaded.predict(x), expected, atol=0.015, rtol=0.015)
    assert len(loaded.layers) == 1


def test_inference_snapshot_does_not_keep_training_state_or_share_weights():
    model = Sequential([Dense(4, 5), ReLU(), Dense(5, 2)])
    x = np.ones((3, 4), np.float32)
    model.train(x, np.zeros((3, 2), np.float32), epochs=1, verbose=False)
    exported = InferenceModel.from_sequential(model)
    expected = exported.predict(x)
    assert model.layers[0].training
    model.layers[0].weights[:] = 0
    np.testing.assert_array_equal(exported.predict(x), expected)
    for layer in exported.layers:
        assert not hasattr(layer, "backward")
        assert not hasattr(layer, "_input_cache")


def test_float16_storage_accumulates_in_float32():
    model = Sequential([Dense(2, 1, use_bias=False)])
    model.layers[0].weights[:] = 1000
    exported = InferenceModel.from_sequential(model, quantize="float16")
    np.testing.assert_allclose(
        exported.predict(np.array([[1000, 1000]], np.float32)), [[2e6]]
    )


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize(
    "values", [[0, 0, 0], [3, 3, 3], [-7, -7, -7], [-3, 1, 6], [1, 2, 3]]
)
def test_asymmetric_int8_handles_signed_and_constant_tensors(values, per_channel):
    weights = np.array([values, values], np.float32)
    params = _compute_int8_params(weights, symmetric=False, per_channel=per_channel)
    reconstructed = _dequantize_from_int8(_quantize_to_int8(weights, params), params)
    assert np.all(np.isfinite(reconstructed))
    np.testing.assert_allclose(reconstructed, weights, atol=np.max(params.scale) * 0.51)


def test_per_channel_int8_preserves_small_output_channels_and_scales(tmp_path):
    model = Sequential([Dense(4, 3, use_bias=False)])
    model.layers[0].weights[:] = np.array(
        [[100, 0.01, 0], [-50, -0.02, 0], [25, 0.03, 0], [-20, -0.04, 0]]
    )
    x = np.eye(4, dtype=np.float32)
    tensor = InferenceModel.from_sequential(model, quantize="int8")
    channel = InferenceModel.from_sequential(model, quantize="int8", per_channel=True)
    assert np.max(np.abs(channel.predict(x)[:, 1] - model.predict(x)[:, 1])) < 0.001
    assert np.max(np.abs(tensor.predict(x)[:, 1] - model.predict(x)[:, 1])) > 0.01
    channel.save(tmp_path / "per_channel")
    loaded = InferenceModel.load(tmp_path / "per_channel")
    np.testing.assert_array_equal(loaded.predict(x), channel.predict(x))
    assert loaded.memory_bytes == channel.memory_bytes
    assert (
        channel.memory_bytes == 12 + 12 + 4
    )  # Packed weights, scales, scalar zero point.


def test_batchnorm_folding_and_channel_compaction_preserve_predictions():
    rng = np.random.default_rng(19)
    model = Sequential(
        [
            Dense(6, 10, use_bias=False),
            BatchNorm(10),
            ReLU(),
            Dropout(0.4),
            Dense(10, 2),
        ]
    )
    bn = model.layers[1]
    bn.running_mean[:] = rng.normal(size=(1, 10))
    bn.running_var[:] = rng.uniform(0.3, 2, size=(1, 10))
    bn.gamma[:] = rng.normal(size=(1, 10))
    bn.beta[:] = rng.normal(size=(1, 10))
    model.layers[-1].weights[2:8] = 0
    x = rng.normal(size=(12, 6)).astype(np.float32)
    expected = model.predict(x)
    original = model.layers[0].weights.copy()
    exported = InferenceModel.from_sequential(model, optimize=True)
    assert len(exported.layers) == 2
    assert exported.layers[0].weights.shape == (6, 4)
    assert exported.layers[1].weights.shape == (4, 2)
    np.testing.assert_allclose(exported.predict(x), expected, atol=2e-6)
    np.testing.assert_array_equal(original, model.layers[0].weights)
    assert exported.memory_bytes < InferenceModel.from_sequential(model).memory_bytes


def test_batchnorm_after_nonlinearity_is_not_folded():
    model = Sequential(
        [Dense(3, 4, activation=SigmoidActivation()), BatchNorm(4), Dense(4, 2)]
    )
    x = np.ones((5, 3), np.float32)
    optimized = InferenceModel.from_sequential(model, optimize=True)
    assert len(optimized.layers) == 3
    np.testing.assert_allclose(optimized.predict(x), model.predict(x), atol=1e-6)


def test_softmax_channels_are_not_pruned():
    model = Sequential([Dense(3, 5), Softmax(), Dense(5, 1)])
    model.layers[-1].weights[1:] = 0
    exported = InferenceModel.from_sequential(model, optimize=True)
    assert exported.layers[0].weights.shape[1] == 5
    x = np.ones((3, 3), np.float32)
    np.testing.assert_allclose(exported.predict(x), model.predict(x), atol=1e-6)


def test_zero_hidden_outputs_compact_even_with_nonzero_outgoing_weights():
    model = Sequential([Dense(3, 6), ReLU(), Dense(6, 2)])
    model.layers[0].weights[:, 1:4] = 0
    model.layers[0].bias[:, 1:4] = 0
    x = np.ones((4, 3), np.float32)
    exported = InferenceModel.from_sequential(model, optimize=True)
    assert exported.layers[0].weights.shape == (3, 3)
    np.testing.assert_allclose(exported.predict(x), model.predict(x), atol=1e-6)


def test_pruning_threshold_is_explicit_and_validated():
    model = Sequential([Dense(3, 4), ReLU(), Dense(4, 2)])
    model.layers[-1].weights[0] = 1e-5
    exact = InferenceModel.from_sequential(model, optimize=True)
    lossy = InferenceModel.from_sequential(model, optimize=True, prune_threshold=1e-4)
    assert lossy.memory_bytes < exact.memory_bytes
    for invalid in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError, match="prune_threshold"):
            InferenceModel.from_sequential(
                model, optimize=True, prune_threshold=invalid
            )
    with pytest.raises(ValueError, match="optimize"):
        InferenceModel.from_sequential(model, prune_threshold=0.1)


def test_flatten_input_and_generic_activation_export(tmp_path):
    model = Sequential([Input((2, 3)), Flatten(), Dense(6, 2), Activation("sigmoid")])
    x = np.ones((3, 2, 3), np.float32)
    inference = InferenceModel.from_sequential(model, quantize="int8", optimize=True)
    inference.save(tmp_path / "flat")
    np.testing.assert_allclose(
        InferenceModel.load(tmp_path / "flat").predict(x), model.predict(x), atol=0.015
    )


def _compile_and_predict(tmp_path, inference, x):
    compiler = shutil.which("cc") or shutil.which("gcc")
    if not compiler:
        pytest.skip("A C99 compiler is required")
    report = inference.export_c(tmp_path / "model", name="tiny_net")
    library = tmp_path / "model.so"
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O2",
            "-shared",
            "-fPIC",
            report["source_path"],
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    runtime = ctypes.CDLL(str(library))
    predict = runtime.tiny_net_predict
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    predict.argtypes = [pointer, pointer, pointer]
    predict.restype = None
    results = []
    # Guard values catch writes past the declared scratch/output sizes.
    for row in x:
        scratch = np.full(report["scratch_floats"] + 2, 12345, np.float32)
        output = np.full(report["output_size"] + 2, 12345, np.float32)
        predict(
            np.ascontiguousarray(row, dtype=np.float32), output[1:-1], scratch[1:-1]
        )
        np.testing.assert_array_equal(scratch[[0, -1]], [12345, 12345])
        np.testing.assert_array_equal(output[[0, -1]], [12345, 12345])
        results.append(output[1:-1].copy())
    return np.array(results), report


@pytest.mark.parametrize("precision", [None, "int8"])
@pytest.mark.parametrize(
    "activation", [ReLU(), Sigmoid(), Tanh(), Softmax(), LeakyReLU(0.2), ELU(1.3)]
)
def test_compiled_c_matches_python_runtime(tmp_path, precision, activation):
    rng = np.random.default_rng(100)
    model = Sequential(
        [
            Dense(3, 5),
            BatchNorm(5),
            activation,
            Dense(5, 2, activation=SigmoidActivation()),
        ]
    )
    inference = InferenceModel.from_sequential(
        model, quantize=precision, per_channel=True, optimize=True
    )
    x = rng.normal(size=(6, 3)).astype(np.float32)
    actual, report = _compile_and_predict(tmp_path, inference, x)
    np.testing.assert_allclose(actual, inference.predict(x), atol=1e-6, rtol=1e-5)
    assert report["scratch_bytes"] == report["scratch_floats"] * 4
    assert report["weight_bytes"] == inference.memory_bytes
    assert (
        json.loads((tmp_path / "model.report.json").read_text())["multiply_accumulates"]
        == 25
    )


def test_compiled_c_unfolded_batchnorm_and_flatten(tmp_path):
    model = Sequential(
        [Flatten(), Dense(3, 5, use_bias=False), Sigmoid(), BatchNorm(5), Dense(5, 2)]
    )
    inference = InferenceModel.from_sequential(model, quantize="int8")
    x = np.ones((3, 3), np.float32)
    actual, report = _compile_and_predict(tmp_path, inference, x)
    np.testing.assert_allclose(actual, inference.predict(x), atol=1e-6)
    assert report["input_size"] == 3


def test_c_budgets_reject_before_writing_files(tmp_path):
    inference = InferenceModel.from_sequential(
        Sequential([Dense(8, 16), ReLU(), Dense(16, 2)]), quantize="int8", optimize=True
    )
    path = tmp_path / "over_budget"
    with pytest.raises(ValueError, match="Weight budget exceeded"):
        inference.export_c(path, max_weight_bytes=1)
    with pytest.raises(ValueError, match="Scratch budget exceeded"):
        inference.export_c(path, max_scratch_bytes=1)
    assert list(tmp_path.iterdir()) == []
    report = inference.export_c(path)
    inference.export_c(
        path,
        max_weight_bytes=report["weight_bytes"],
        max_scratch_bytes=report["scratch_bytes"],
    )


@pytest.mark.parametrize("name", ["bad-name", "1model", "float", "_reserved", "name\n"])
def test_c_rejects_invalid_symbol_names(tmp_path, name):
    inference = InferenceModel.from_sequential(Sequential([Dense(2, 1)]))
    with pytest.raises(ValueError, match="identifier"):
        inference.export_c(tmp_path / "invalid", name=name)


def test_invalid_quantization_is_rejected():
    with pytest.raises(ValueError, match="dtype"):
        QuantizationConfig(dtype="bad")


@pytest.mark.parametrize("activation", [LeakyReLUActivation(0.3), ELUActivation(1.7)])
def test_training_model_persistence_preserves_activation_parameters(
    tmp_path, activation
):
    model = Sequential(
        [
            Input((2, 2)),
            Flatten(),
            Dense(4, 3, activation=activation),
            Activation(activation),
        ]
    )
    x = np.ones((3, 2, 2), np.float32)
    model.save(tmp_path / "training")
    loaded = InferenceModel.from_file(tmp_path / "training", quantize="float16")
    np.testing.assert_allclose(
        loaded.predict(x), model.predict(x), atol=0.01, rtol=0.01
    )


def test_loads_version_1_wrapped_artifacts(tmp_path):
    config = {
        "artifact_type": "inference_model",
        "format_version": "1.0",
        "layers": [
            {
                "runtime": "wrapped",
                "class": "Dense",
                "in_features": 2,
                "out_features": 1,
                "use_bias": False,
                "weights_key": "weights",
                "activation": {"name": "relu"},
            }
        ],
    }
    (tmp_path / "legacy.json").write_text(json.dumps(config))
    np.savez(tmp_path / "legacy.npz", weights=np.array([[1], [-2]], np.float32))
    loaded = InferenceModel.load(tmp_path / "legacy")
    np.testing.assert_array_equal(loaded.predict([[1, 1], [3, 1]]), [[0], [1]])


@pytest.mark.parametrize("bad_weights", [np.nan, np.inf, 1e6])
def test_nonfinite_or_unrepresentable_fp16_export_is_rejected(bad_weights):
    model = Sequential([Dense(2, 1)])
    model.layers[0].weights[:] = bad_weights
    with pytest.raises(ValueError, match="finite|representable"):
        InferenceModel.from_sequential(model, quantize="float16")


def test_int8_weight_expansion_stays_bounded():
    import tracemalloc
    from neura_command.inference.quantization import (
        QuantizedDense,
        Int8QuantizationParams,
    )

    weights = np.ones((2048, 512), np.int8)
    layer = QuantizedDense(
        weights, None, "int8", {"weights": Int8QuantizationParams(0.5, 0, 0, 1)}
    )
    x = np.ones((1, 2048), np.float32)
    tracemalloc.start()
    try:
        result = layer.forward(x)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    np.testing.assert_allclose(result, 1024)
    assert peak < weights.size * 2  # A full float32 matrix alone would use 4x.
