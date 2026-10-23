"""Portable C99 inference with constant weights and caller-owned scratch RAM."""

import json
import re
from pathlib import Path

import numpy as np

from neura_command.backend import to_numpy
from neura_command.inference.quantization import (
    QuantizedActivation,
    QuantizedBatchNorm,
    QuantizedDense,
    QuantizedDropout,
    QuantizedFlatten,
)

_C_KEYWORDS = set(
    "auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while _Bool _Complex _Imaginary".split()
)


def _float_literal(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("C export requires finite parameters")
    literal = format(value, ".9g")
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return literal + "f"


def _activation_code(activation, width):
    if activation is None or activation.activation_type == "linear":
        return []
    kind = activation.activation_type
    alpha = _float_literal(
        activation.alpha
        if activation.alpha is not None
        else (0.01 if kind == "leaky_relu" else 1.0)
    )
    if kind == "softmax":
        return [
            "    {",
            "        float maximum = a[0];",
            f"        for (size_t j = 1; j < {width}; ++j) if (a[j] > maximum) maximum = a[j];",
            "        float total = 0.0f;",
            f"        for (size_t j = 0; j < {width}; ++j) {{ a[j] = expf(a[j] - maximum); total += a[j]; }}",
            f"        for (size_t j = 0; j < {width}; ++j) a[j] /= total;",
            "    }",
        ]
    expressions = {
        "relu": "a[j] > 0.0f ? a[j] : 0.0f",
        "tanh": "tanhf(a[j])",
        "leaky_relu": f"a[j] > 0.0f ? a[j] : {alpha} * a[j]",
        "elu": f"a[j] > 0.0f ? a[j] : {alpha} * expm1f(a[j])",
    }
    if kind == "sigmoid":
        return [
            f"    for (size_t j = 0; j < {width}; ++j) {{",
            "        float z = expf(-fabsf(a[j]));",
            "        a[j] = a[j] >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);",
            "    }",
        ]
    if kind not in expressions:
        raise ValueError(f"Unsupported C activation: {kind}")
    return [f"    for (size_t j = 0; j < {width}; ++j) a[j] = {expressions[kind]};"]


def export_c(
    model, path, *, name="neura_model", max_weight_bytes=None, max_scratch_bytes=None
):
    """Export a float32 or weight-only INT8 model as .c, .h, and .report.json.

    No allocator or Python runtime is used by generated code. Weight bytes
    count constant parameter arrays, excluding executable code, alignment, and
    libm. Scratch bytes are exact for the generated two-buffer implementation;
    caller input/output arrays and scalar call-stack overhead are additional.
    """
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in _C_KEYWORDS:
        raise ValueError(
            "name must be a non-reserved C identifier starting with a letter"
        )
    base = Path(path)
    if base.suffix in {".c", ".h"}:
        base = base.with_suffix("")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", base.name):
        raise ValueError(
            "C export filename may contain only letters, numbers, '_', '-', and '.'"
        )
    for limit in (max_weight_bytes, max_scratch_bytes):
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("Memory budgets must be non-negative integer byte counts")

    dense_layers = [
        layer for layer in model.layers if isinstance(layer, QuantizedDense)
    ]
    if not dense_layers:
        raise ValueError("C export requires at least one Dense inference layer")
    input_size = dense_layers[0].weights.shape[0]
    if input_size < 1:
        raise ValueError("C export requires a positive input size")
    width = input_size
    max_width = width
    weight_bytes = 0
    macs = 0
    constants = []
    body = []

    def array(identifier, values, dtype="float"):
        nonlocal weight_bytes
        values = np.asarray(to_numpy(values))
        if not np.all(np.isfinite(values)):
            raise ValueError("C export requires finite parameters")
        flat = values.reshape(-1)
        element_size = 1 if dtype == "int8_t" else 4
        weight_bytes += flat.size * element_size
        literals = [
            str(int(value)) if dtype != "float" else _float_literal(value)
            for value in flat
        ]
        rows = [
            "    " + ", ".join(literals[i : i + 12])
            for i in range(0, len(literals), 12)
        ]
        constants.append(
            f"static const {dtype} {identifier}[{flat.size}] = {{\n"
            + ",\n".join(rows)
            + "\n};"
        )
        return identifier

    for index, layer in enumerate(model.layers):
        prefix = f"{name}_layer_{index}"
        if isinstance(layer, QuantizedDense):
            if layer.dtype == "float16":
                raise ValueError(
                    "C99 export supports float32 or int8 storage; re-export with quantize='int8' or None"
                )
            inputs, outputs = layer.weights.shape
            if inputs != width or outputs < 1:
                raise ValueError(
                    "Dense layer dimensions do not form a valid sequential network"
                )
            weights = array(
                prefix + "_weights",
                layer.weights,
                "int8_t" if layer.dtype == "int8" else "float",
            )
            bias = (
                array(prefix + "_bias", layer.bias) if layer.bias is not None else None
            )
            weight_expr = f"{weights}[k * {outputs} + j]"
            if layer.dtype == "int8":
                params = layer.quant_params["weights"]
                scales = np.asarray(to_numpy(params.scale)).reshape(-1)
                zeros = np.asarray(to_numpy(params.zero_point)).reshape(-1)
                if scales.size not in {1, outputs} or zeros.size not in {1, outputs}:
                    raise ValueError(
                        "INT8 quantization parameters must be scalar or per output channel"
                    )
                scale = array(prefix + "_scale", scales)
                zero = array(prefix + "_zero", zeros, "int32_t")
                weight_expr = f"((float){weight_expr} - (float){zero}[{'j' if zeros.size > 1 else '0'}]) * {scale}[{'j' if scales.size > 1 else '0'}]"
            body.extend(
                [
                    f"    for (size_t j = 0; j < {outputs}; ++j) {{",
                    f"        float sum = {bias + '[j]' if bias else '0.0f'};",
                    f"        for (size_t k = 0; k < {inputs}; ++k) sum += a[k] * ({weight_expr});",
                    "        b[j] = sum;",
                    "    }",
                    "    { float *temporary = a; a = b; b = temporary; }",
                ]
            )
            width = outputs
            macs += inputs * outputs
            max_width = max(max_width, width)
            body.extend(_activation_code(layer.activation, width))
        elif isinstance(layer, QuantizedActivation):
            body.extend(_activation_code(layer, width))
        elif isinstance(layer, QuantizedBatchNorm):
            if layer.gamma.size != width:
                raise ValueError("BatchNorm feature count does not match the network")
            scale = to_numpy(layer.gamma).astype(np.float32) / np.sqrt(
                to_numpy(layer.running_var).astype(np.float32) + layer.eps
            )
            offset = to_numpy(layer.beta) - to_numpy(layer.running_mean) * scale
            scales = array(prefix + "_scale", scale)
            offsets = array(prefix + "_offset", offset)
            body.append(
                f"    for (size_t j = 0; j < {width}; ++j) a[j] = a[j] * {scales}[j] + {offsets}[j];"
            )
        elif isinstance(layer, (QuantizedDropout, QuantizedFlatten)):
            pass  # C accepts one already flattened sample.
        else:
            raise ValueError(
                f"Unsupported C inference layer: {type(layer).__name__}. Re-export from a Sequential model."
            )

    scratch_floats = 2 * max_width
    scratch_bytes = scratch_floats * 4
    if max_weight_bytes is not None and weight_bytes > max_weight_bytes:
        raise ValueError(
            f"Weight budget exceeded: {weight_bytes} bytes required, limit {max_weight_bytes}"
        )
    if max_scratch_bytes is not None and scratch_bytes > max_scratch_bytes:
        raise ValueError(
            f"Scratch budget exceeded: {scratch_bytes} bytes required, limit {max_scratch_bytes}"
        )

    header_path = Path(str(base) + ".h")
    source_path = Path(str(base) + ".c")
    report_path = Path(str(base) + ".report.json")
    macro = name.upper()
    header = f"""/* Generated by NeuraCommand. C99, float32 inputs/outputs/accumulation.
 * Caller supplies scratch RAM disjoint from input/output. No dynamic allocation.
 * Constant arrays may be placed in flash by the target linker.
 */
#ifndef {macro}_H
#define {macro}_H
#define {macro}_INPUT_SIZE {input_size}
#define {macro}_OUTPUT_SIZE {width}
#define {macro}_WEIGHT_BYTES {weight_bytes}
#define {macro}_SCRATCH_FLOATS {scratch_floats}
#define {macro}_SCRATCH_BYTES {scratch_bytes}
#ifdef __cplusplus
extern "C" {{
#endif
void {name}_predict(const float *input, float *output, float *scratch);
#ifdef __cplusplus
}}
#endif
#endif
"""
    source = (
        f'#include "{header_path.name}"\n#include <stddef.h>\n#include <stdint.h>\n#include <math.h>\n\n'
        + "\n\n".join(constants)
        + f"\n\nvoid {name}_predict(const float *input, float *output, float *scratch) {{\n"
        + f"    float *a = scratch;\n    float *b = scratch + {max_width};\n"
        + f"    for (size_t j = 0; j < {input_size}; ++j) a[j] = input[j];\n"
        + "\n".join(body)
        + f"\n    for (size_t j = 0; j < {width}; ++j) output[j] = a[j];\n}}\n"
    )
    report = {
        "name": name,
        "input_size": input_size,
        "output_size": width,
        "weight_bytes": int(weight_bytes),
        "scratch_bytes": scratch_bytes,
        "scratch_floats": scratch_floats,
        "input_bytes": input_size * 4,
        "output_bytes": width * 4,
        "multiply_accumulates": macs,
        "layers": len(model.layers),
        "quantized": model.metadata.get("quantized", "none"),
        "metadata": model.metadata,
        "source_path": str(source_path),
        "header_path": str(header_path),
        "report_path": str(report_path),
    }
    base.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
