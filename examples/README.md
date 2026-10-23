# Reproducible demos

Install the project from the repository root, then choose a demo:

| Script | Demonstrates | Default output directory |
| --- | --- | --- |
| `embedded_demo.py` | Tiny classifier, BatchNorm folding, early stopping, INT8/C export | `build/embedded-demo` |
| `regression_demo.py` | Two-output regression, correctly normalized MSE, INT8 error measurement | `build/regression-demo` |
| `cli_demo.py` | Separate create/train/info/export/predict CLI processes and JSON reports | `build/cli-demo` |

```sh
python -m pip install -e .
python examples/embedded_demo.py --seed 42 --verify-c
python examples/regression_demo.py --seed 42 --verify-c
python examples/cli_demo.py --seed 42 --verify-c
```

Each script writes one JSON metrics object to stdout and saves the same object as
`metrics.json`. Inputs and targets are generated locally; no dataset download or
extra machine-learning framework is needed. Every run uses separate training,
validation, and held-out test data. These are synthetic workflow examples, not
real application benchmarks.

## Adjust a run

```sh
python examples/embedded_demo.py --device cpu --seed 7 --epochs 40 \
  --samples 320 --hidden 8 --batch-size 16 --output-dir build/small-demo \
  --max-weight-bytes 512 --max-scratch-bytes 128 --verify-c
```

All scripts accept `--device cpu|cuda|gpu`, `--seed`, `--epochs`, `--samples`,
`--hidden`, `--batch-size`, `--output-dir`, `--max-weight-bytes`, and
`--max-scratch-bytes`. At least eight samples are required. Defaults keep the
network below a 1,024-byte parameter budget and a 256-byte scratch budget.

For GPU runs, first install exactly one CuPy build matching the driver/toolkit
(see the project README), then select `--device cuda`. Seeds control initialization,
validation splitting, sample order, and dropout on the selected backend. Exact
bitwise agreement across different hardware/backends is not guaranteed.

## Inspect the outputs

- `train_*`, `validation_*`, and `test_*` CSV files make the data inspectable.
- `trained.json`/`trained.npz` can be loaded for further training on CPU or CUDA.
- `inference.json`/`inference.npz` contain the optimized INT8 inference model.
- `history.json` records loss, validation loss, and learning rate by epoch.
- `model.c`, `model.h`, and `model.report.json` describe the C99 deployment.
- `metrics.json` compares original/INT8 accuracy or MSE and reports storage/RAM.

The CLI demo also saves its architecture, pooled training CSVs, inference info,
and prediction JSON. Its seeded CLI validation split uses the training pool;
held-out test samples are used only for final reporting. API demos restore the
best validation weights before reporting test results.

`--verify-c` requires a native POSIX C99 compiler (`cc` by default, or `--cc gcc`).
It builds a temporary shared library, runs every held-out sample through it, and
compares predictions with Python INT8 inference. The metrics include the maximum
absolute difference. Temporary binaries are cleaned up. This is a host check;
measure firmware size, stack use, and latency on the actual embedded target.

Run output directories can be reused, replacing that demo's generated files.
Use distinct `--output-dir` values when comparing configurations.
