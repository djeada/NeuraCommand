"""Shared data, export, and native-C checks for the runnable demos."""

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from app.commands.base_command import BaseCommand
from neura_command import InferenceModel


def demo_parser(description, *, output_dir, epochs=60, samples=480):
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    parser.add_argument("--seed", type=BaseCommand.seed, default=42)
    parser.add_argument("--epochs", type=BaseCommand.positive_int, default=epochs)
    parser.add_argument("--batch-size", type=BaseCommand.positive_int, default=32)
    parser.add_argument("--hidden", type=BaseCommand.positive_int, default=12)
    parser.add_argument("--samples", type=BaseCommand.positive_int, default=samples)
    parser.add_argument("--output-dir", type=Path, default=Path(output_dir))
    parser.add_argument(
        "--max-weight-bytes", type=BaseCommand.nonnegative_int, default=1024
    )
    parser.add_argument(
        "--max-scratch-bytes", type=BaseCommand.nonnegative_int, default=256
    )
    parser.add_argument(
        "--verify-c",
        action="store_true",
        help="Compile and execute exported C on the host and compare predictions.",
    )
    parser.add_argument(
        "--cc", default="cc", help="Native C99 compiler for --verify-c (default: cc)."
    )
    return parser


def parse_demo(parser, argv=None):
    args = parser.parse_args(argv)
    if args.samples < 8:
        parser.error(
            "--samples must be at least 8 for non-empty train/validation/test splits"
        )
    return args


def classification_data(seed, samples):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, 6)).astype(np.float32)
    teacher = rng.normal(size=(6, 3)).astype(np.float32)
    labels = (x @ teacher).argmax(axis=1)
    return x, np.eye(3, dtype=np.float32)[labels]


def regression_data(seed, samples):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(samples, 4)).astype(np.float32)
    weights = np.array([[0.7, -0.2], [-0.5, 0.3], [0.2, 0.8], [0.4, -0.6]], np.float32)
    y = x @ weights + 0.05 * np.sin(x[:, :2] * 3)
    return x, y.astype(np.float32)


def split_data(x, y):
    train_end = len(x) * 3 // 5
    validation_end = len(x) * 4 // 5
    return (
        (x[:train_end], y[:train_end]),
        (x[train_end:validation_end], y[train_end:validation_end]),
        (x[validation_end:], y[validation_end:]),
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def save_data(directory, splits):
    directory.mkdir(parents=True, exist_ok=True)
    for name, (x, y) in zip(("train", "validation", "test"), splits):
        np.savetxt(directory / f"{name}_features.csv", x, delimiter=",")
        np.savetxt(directory / f"{name}_targets.csv", y, delimiter=",")


def verify_c(inference, report, x, compiler="cc"):
    executable = shutil.which(compiler)
    if executable is None:
        raise ValueError(f"Native C compiler not found: {compiler}")
    x = np.ascontiguousarray(x.reshape(len(x), -1), dtype=np.float32)
    if x.shape[1] != report["input_size"]:
        raise ValueError(
            "Verification features do not match the generated C input size"
        )
    with tempfile.TemporaryDirectory(prefix="neura-c-check-") as directory:
        library_path = Path(directory) / "model.so"
        command = [
            executable,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-O2",
            "-shared",
            "-fPIC",
            str(report["source_path"]),
            "-o",
            str(library_path),
            "-lm",
        ]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise ValueError("Native C compilation failed: " + compiled.stderr.strip())
        library = ctypes.CDLL(str(library_path))
        predict = getattr(library, report["name"] + "_predict")
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        predict.argtypes = [pointer, pointer, pointer]
        predict.restype = None
        scratch = np.empty(report["scratch_floats"], np.float32)
        output = np.empty((len(x), report["output_size"]), np.float32)
        for row, result in zip(x, output):
            predict(row, result, scratch)
        expected = inference.predict(x)
        if not np.allclose(output, expected, atol=2e-5, rtol=2e-4):
            raise ValueError("Compiled C predictions differ from Python inference")
        return {
            "samples": len(x),
            "max_absolute_error": float(np.max(np.abs(output - expected))),
        }


def export_and_measure(model, history, args, test, task):
    directory = args.output_dir
    model.save(directory / "trained")
    write_json(directory / "history.json", history)
    inference = InferenceModel.from_sequential(
        model, quantize="int8", per_channel=True, optimize=True
    )
    inference.save(directory / "inference")
    report = inference.export_c(
        directory / "model",
        max_weight_bytes=args.max_weight_bytes,
        max_scratch_bytes=args.max_scratch_bytes,
    )
    x, y = test
    original, packed = model.predict(x), inference.predict(x)
    metrics = {
        "task": task,
        "seed": args.seed,
        "device": model.device,
        "epochs_completed": len(history["loss"]),
        "test_samples": len(x),
        "original_parameter_bytes": InferenceModel.from_sequential(model).memory_bytes,
        "embedded_weight_bytes": report["weight_bytes"],
        "scratch_bytes": report["scratch_bytes"],
        "original_loss": float(model.loss.compute(y, original)),
        "int8_loss": float(model.loss.compute(y, packed)),
        "max_quantization_error": float(np.max(np.abs(original - packed))),
    }
    if task == "classification":
        metrics["original_accuracy"] = float(np.mean(original.argmax(1) == y.argmax(1)))
        metrics["int8_accuracy"] = float(np.mean(packed.argmax(1) == y.argmax(1)))
    if args.verify_c:
        metrics["c_verification"] = verify_c(inference, report, x, args.cc)
    write_json(directory / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics
