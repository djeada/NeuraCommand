"""Exercise create/train/info/export/predict in separate CLI processes on CPU/CUDA."""

import json
import subprocess
import sys

import numpy as np

from neura_command import InferenceModel
from demo_utils import (
    demo_parser,
    parse_demo,
    classification_data,
    split_data,
    save_data,
    verify_c,
    write_json,
)


def run_cli(*arguments, json_output=False):
    result = subprocess.run(
        [sys.executable, "-m", "neura_command", *map(str, arguments)],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError("CLI command failed: " + result.stderr.strip())
    return json.loads(result.stdout) if json_output else None


def main(argv=None):
    parser = demo_parser(__doc__, output_dir="build/cli-demo", epochs=60, samples=480)
    args = parse_demo(parser, argv)
    try:
        directory = args.output_dir
        splits = split_data(*classification_data(args.seed, args.samples))
        save_data(directory, splits)
        # Pool train/validation here; the CLI makes its own seeded validation split.
        x = np.concatenate((splits[0][0], splits[1][0]))
        y = np.concatenate((splits[0][1], splits[1][1]))
        np.savetxt(directory / "features.csv", x, delimiter=",")
        np.savetxt(directory / "targets.csv", y, delimiter=",")
        write_json(
            directory / "architecture.json",
            {
                "layers": [
                    {"type": "dense", "in_features": 6, "out_features": args.hidden},
                    {"type": "batchnorm", "num_features": args.hidden},
                    {"type": "relu"},
                    {"type": "dense", "in_features": args.hidden, "out_features": 3},
                    {"type": "softmax"},
                ],
                "optimizer": {"class": "Adam", "learning_rate": 0.02},
                "loss": "CrossEntropyLoss",
            },
        )
        run_cli(
            "create",
            "--architecture",
            directory / "architecture.json",
            "--output",
            directory / "initial",
            "--device",
            args.device,
            "--seed",
            args.seed,
            "--quiet",
        )
        run_cli(
            "train",
            "--model",
            directory / "initial",
            "--features-csv",
            directory / "features.csv",
            "--targets-csv",
            directory / "targets.csv",
            "--output",
            directory / "trained",
            "--device",
            args.device,
            "--seed",
            args.seed,
            "--epochs",
            args.epochs,
            "--batch-size",
            args.batch_size,
            "--validation-split",
            "0.25",
            "--early-stopping",
            "12",
            "--save-history",
            directory / "history.json",
            "--quiet",
        )
        run_cli(
            "export",
            "--model",
            directory / "trained",
            "--output",
            directory / "inference",
            "--quantize",
            "int8",
            "--per-channel",
            "--max-weight-bytes",
            args.max_weight_bytes,
            "--quiet",
        )
        report = run_cli(
            "export",
            "--model",
            directory / "trained",
            "--output",
            directory / "model",
            "--quantize",
            "int8",
            "--per-channel",
            "--format",
            "c",
            "--json",
            "--max-weight-bytes",
            args.max_weight_bytes,
            "--max-scratch-bytes",
            args.max_scratch_bytes,
            json_output=True,
        )
        info = run_cli(
            "info",
            "--model",
            directory / "inference",
            "--format",
            "json",
            json_output=True,
        )
        write_json(directory / "info.json", info)
        predictions = run_cli(
            "predict",
            "--model",
            directory / "inference",
            "--input-csv",
            directory / "test_features.csv",
            "--device",
            args.device,
            "--output-mode",
            "json",
            "--quiet",
            json_output=True,
        )
        write_json(directory / "predictions.json", predictions)
        values = np.asarray(predictions["predictions"])
        metrics = {
            "task": "classification",
            "workflow": "cli",
            "seed": args.seed,
            "device": args.device,
            "test_samples": len(values),
            "int8_accuracy": float(np.mean(values.argmax(1) == splits[2][1].argmax(1))),
            "embedded_weight_bytes": report["weight_bytes"],
            "scratch_bytes": report["scratch_bytes"],
        }
        if args.verify_c:
            metrics["c_verification"] = verify_c(
                InferenceModel.load(directory / "inference"),
                report,
                splits[2][0],
                args.cc,
            )
        write_json(directory / "metrics.json", metrics)
        print(json.dumps(metrics, indent=2))
        return metrics
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
