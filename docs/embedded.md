# Embedded deployment and device execution

NeuraCommand trains small dense networks on a NumPy CPU backend or an optional
CuPy CUDA backend. Deploy on a Python-capable edge device with an inference
artifact, or export a C99 function for an embedded CPU without Python/NumPy.
Supported export layers are Dense, BatchNorm, the built-in activations, Flatten,
Input, and Dropout. C input is one flattened sample.

## Reproduce the workflow

From an installed checkout:

```sh
python examples/embedded_demo.py --device cpu
# With matching CuPy, CUDA toolkit, and an NVIDIA driver:
python examples/embedded_demo.py --device cuda --output-dir build/cuda-demo
```

The example trains a 6 → 12 → 3 classifier on synthetic data with separate
training, validation, and test splits. It holds out 96 test samples and writes
CSV inputs/targets, training history, metrics, a portable training model, an INT8
inference artifact, C source/header, and a resource report. It requests a 1,024-byte
parameter budget and 256-byte scratch budget. Its synthetic accuracy is a workflow check, not an application
benchmark. Use application-specific validation data and target measurements for
shipping decisions.

Pass `--verify-c` to compile and execute the generated function on all held-out
test inputs and record its maximum absolute difference from Python INT8 inference.
`--seed`, `--epochs`, `--hidden`, `--samples`, and `--batch-size` control the run.
The [demo guide](../examples/README.md) also covers a two-output regression demo
and a complete CLI workflow demo.

## Export for firmware

```sh
neura_command export --model trained --output firmware/model \
  --format c --quantize int8 --per-channel \
  --max-weight-bytes 16384 --max-scratch-bytes 2048
```

The output is `model.c`, `model.h`, and `model.report.json`. Export rejects an
oversized model before writing files. Check the process exit code; export errors
are nonzero. A typical caller looks like:

```c
#include "model.h"

void run_sample(const float input[NEURA_MODEL_INPUT_SIZE],
                float output[NEURA_MODEL_OUTPUT_SIZE]) {
    float scratch[NEURA_MODEL_SCRATCH_FLOATS];
    neura_model_predict(input, output, scratch);
}
```

The caller owns scratch memory, so separate scratch arrays allow concurrent calls.
Scratch must not overlap input/output. Use `--name sensor_net` to change the C
symbol and macro prefix. For hosts with small stacks, allocate the scratch array
in application-owned static memory instead. Compile `model.c` as C99 and link
the target's C math library (`-lm` where required). A host compile check is:

```sh
cc -std=c99 -Wall -Wextra -Werror -pedantic -Os -c firmware/model.c -o firmware/model.o
```

The generated function has no allocator, mutable global state, or dependency on
the training library. Weights use `static const` arrays; place read-only data in
flash with the target's linker configuration. Confirm the linker map on the
actual microcontroller: some targets have a distinct program-memory address
space or need special access intrinsics. Those are not generated automatically.

## What the resource report measures

| Field | Meaning |
| --- | --- |
| `weight_bytes` | Constant parameter arrays, including biases and INT8 scales/zero points |
| `scratch_bytes` | Exact caller buffer size for two float32 activation buffers |
| `input_bytes`, `output_bytes` | Caller-owned buffers for one sample |
| `multiply_accumulates` | Dense dot-product MAC count for one sample |
| `layers` | Layers after the requested export optimizations |

The report excludes machine code, linker alignment, runtime math library code,
and scalar stack overhead. Inspect the firmware map and measure latency and
stack use on your target. Python `InferenceModel.memory_bytes` counts parameter
storage, including quantization scales, and excludes Python objects and temporary
array allocations; it is not a total process RAM measurement. JSON+NPZ file size
also includes container metadata.

C99 uses float32 inputs, outputs, and accumulation. INT8 refers to **weight-only**
storage, not an integer-only operator pipeline or a vendor-optimized DSP kernel.
C reads each packed weight directly without expanding a full floating-point
matrix. Python/CUDA inference processes FP16/INT8 weights in blocks of at most
32 output channels to bound weight expansion. C99 accepts float32 or INT8 weights;
use Python artifacts for float16 storage.

## Reduce a model safely

CLI export enables optimizations by default. The Python API requires
`InferenceModel.from_sequential(model, optimize=True)` explicitly.

1. Dropout/Input identities are removed from the exported graph.
2. BatchNorm immediately following a linear Dense layer is folded into its
   weights and bias using running statistics. It is not folded across a nonlinear
   activation. Export operates on independent copies.
3. Elementwise activations can be fused into Dense execution.
4. Hidden channels with zero outgoing weights, or provably zero outputs, are
   physically removed from adjacent matrices. Softmax channels are retained
   because changing them would change other channels' probabilities.
5. Optionally, `--prune-threshold 0.001` removes channels whose outgoing weights
   are all at/below that magnitude. This is lossy. Validate with held-out data and
   re-tune the training architecture if accuracy drops.

An exact optimization can still differ by floating-point roundoff. INT8/FP16
quantization trades precision for storage and also needs validation. Per-channel
INT8 adds a scale per output channel and helps when channel magnitudes differ.
Zero/constant weights are supported. Activations remain float32, and calibration
data is not required for this weight-only scheme.

Start with narrow hidden layers. Pruning does not guarantee an arbitrary large
network will meet an embedded budget. Enforce the device's actual budgets and
reduce architecture width or retrain when export fails.

## CPU/CUDA portability and testing

Install the base package for CPU. Add exactly one matching optional extra,
`.[cuda12]` or `.[cuda13]`, for CUDA; see the
[CuPy installation requirements](https://docs.cupy.dev/en/stable/install.html).
`--device cpu|cuda|gpu` is available on create/train/predict. In Python, use
`Sequential(..., device="cuda")`, `model.to("cpu")`, or
`InferenceModel.load("deployment", device="cuda")`. `gpu` normalizes to `cuda`.
GPU selection fails clearly when no usable device is available. In-memory device
transfers preserve optimizer moments and remap their parameter IDs. Saved training
models preserve optimizer configuration, rather than a full optimizer checkpoint.
Saved parameters are CPU-portable arrays. Training batches use float32 and only
one batch is gathered/transferred at a time; loading CSV data still requires the
full host dataset in memory. Prediction results return to CPU as NumPy arrays.

New inference artifacts use format 2.0, which records fused activations,
float32 inference primitives, and per-channel scales. This runtime also reads
version 1 artifacts. Older runtimes reject version 2 instead of silently losing
activation or quantization metadata.

```sh
python -m pytest unit_tests/ -v
```

CPU tests compile the exported C with strict C99 warnings, execute it, compare
predictions against Python across activation/precision variants, and guard the
reported scratch/output boundaries. GPU integration tests cover optimizers,
validation, CPU↔GPU continuation, portable save/load, dropout/losses, and
quantized inference. They skip without working CUDA. The manual **CUDA
integration** workflow requires a Linux self-hosted runner with the `gpu` label,
a driver/toolkit, and a C compiler. It asserts GPU availability before testing,
so a skipped GPU suite cannot count as GPU validation. No GPU hardware validation
is implied by CPU CI.
