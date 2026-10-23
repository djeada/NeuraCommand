"""Train a tiny classifier on CPU/CUDA and export portable embedded C.

Run from an installed checkout:
    python examples/embedded_demo.py --device cpu
    python examples/embedded_demo.py --device cuda --output-dir build/cuda-demo

The data is synthetic. Reported accuracy demonstrates the workflow, not a
real application benchmark. Validate with your own held-out device data.
"""

import argparse
from pathlib import Path

import numpy as np

from neura_command import (
    Sequential,
    Dense,
    BatchNorm,
    ReLU,
    Softmax,
    Adam,
    CrossEntropyLoss,
    InferenceModel,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda", "gpu"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("build/embedded-demo"))
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    np.random.seed(42)
    features = rng.normal(size=(480, 6)).astype(np.float32)
    teacher = rng.normal(size=(6, 3)).astype(np.float32)
    labels = (features @ teacher).argmax(axis=1)
    targets = np.eye(3, dtype=np.float32)[labels]
    model = Sequential(
        [Dense(6, 12), BatchNorm(12), ReLU(), Dense(12, 3), Softmax()],
        optimizer=Adam(learning_rate=0.02),
        loss=CrossEntropyLoss(),
        device=args.device,
    )
    model.train(
        features[:384],
        targets[:384],
        epochs=60,
        batch_size=32,
        verbose=False,
        validation_data=(features[384:], targets[384:]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "trained")
    inference = InferenceModel.from_sequential(
        model, quantize="int8", per_channel=True, optimize=True
    )
    inference.save(args.output_dir / "inference")
    report = inference.export_c(
        args.output_dir / "model", max_weight_bytes=1024, max_scratch_bytes=256
    )
    original = model.predict(features[384:])
    packed = inference.predict(features[384:])
    original_storage = InferenceModel.from_sequential(model).memory_bytes
    print(f"Execution device: {model.device}")
    print(
        f"Held-out accuracy: original {(original.argmax(1) == labels[384:]).mean():.1%}; "
        f"INT8 {(packed.argmax(1) == labels[384:]).mean():.1%}"
    )
    print(f"Original parameter storage: {original_storage} bytes")
    print(
        f"Embedded weights: {report['weight_bytes']} bytes; scratch RAM: {report['scratch_bytes']} bytes"
    )
    print(f"Generated C: {report['source_path']}; header: {report['header_path']}")


if __name__ == "__main__":
    main()
