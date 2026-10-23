"""Inference primitives and weight-only quantization for NumPy/CuPy.

INT8 weights stay packed. Forward passes dequantize at most 32 output
channels at a time; activations and accumulation use float32. This is not
an integer-only runtime. The C exporter reads each packed weight directly.
"""

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

from neura_command.backend import get_array_module, to_numpy, to_device


@dataclass
class QuantizationConfig:
    """Weight storage precision and INT8 quantization granularity."""

    dtype: Literal["float32", "float16", "int8"] = "float16"
    symmetric: bool = True
    per_channel: bool = False

    def __post_init__(self):
        if self.dtype not in {"float32", "float16", "int8"}:
            raise ValueError(f"Unsupported quantization dtype: {self.dtype}")


@dataclass
class Int8QuantizationParams:
    scale: Union[float, np.ndarray]
    zero_point: Union[int, np.ndarray]
    min_val: float
    max_val: float


def _compute_int8_params(weights, symmetric=True, per_channel=False):
    weights = np.asarray(to_numpy(weights), dtype=np.float32)
    if weights.size == 0 or not np.all(np.isfinite(weights)):
        raise ValueError("Quantization requires non-empty, finite weights")
    axis = 0 if per_channel else None
    minimum = weights.min(axis=axis, keepdims=per_channel).astype(np.float64)
    maximum = weights.max(axis=axis, keepdims=per_channel).astype(np.float64)
    if symmetric:
        bound = np.maximum(np.abs(minimum), np.abs(maximum))
        scale = np.maximum(
            np.where(bound > 0, bound / 127.0, 1.0), np.finfo(np.float32).tiny
        ).astype(np.float32)
        zero_point = 0
    else:
        # Include zero, including for constant or one-sided tensors.
        minimum = np.minimum(minimum, 0)
        maximum = np.maximum(maximum, 0)
        span = maximum - minimum
        scale = np.maximum(
            np.where(span > 0, span / 255.0, 1.0), np.finfo(np.float32).tiny
        ).astype(np.float32)
        zero_point = np.clip(np.rint(-128 - minimum / scale), -128, 127).astype(
            np.int32
        )
        zero_point = zero_point if per_channel else int(zero_point)
    return Int8QuantizationParams(
        scale=scale if per_channel else float(scale),
        zero_point=zero_point,
        min_val=float(weights.min()),
        max_val=float(weights.max()),
    )


def _quantize_to_int8(weights, params):
    return np.clip(
        np.rint(weights / params.scale) + params.zero_point, -128, 127
    ).astype(np.int8)


def _dequantize_from_int8(quantized, params):
    xp = get_array_module(quantized)
    return (quantized.astype(xp.float32) - params.zero_point) * params.scale


class QuantizedLayer:
    """Prediction-only primitive: no gradients, optimizer, or input caches."""

    def to(self, device):
        for name, value in vars(self).items():
            if hasattr(value, "shape"):
                setattr(self, name, to_device(value, device))
        for params in getattr(self, "quant_params", {}).values():
            for name in ("scale", "zero_point"):
                value = getattr(params, name)
                if hasattr(value, "shape"):
                    setattr(params, name, to_device(value, device))
        return self

    @property
    def memory_bytes(self):
        total = sum(
            value.nbytes for value in vars(self).values() if hasattr(value, "nbytes")
        )
        for params in getattr(self, "quant_params", {}).values():
            total += sum(
                getattr(params, name).nbytes
                if hasattr(getattr(params, name), "nbytes")
                else 4
                for name in ("scale", "zero_point")
            )
        return int(total)


def _copy_parameter(value, dtype):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(to_numpy(value), dtype=dtype, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"Parameters must be finite and representable as {np.dtype(dtype).name}"
        )
    return result


class QuantizedDense(QuantizedLayer):
    """Packed weights with optional fused activation and bounded dequantization."""

    def __init__(self, weights, bias, dtype, quant_params=None, activation=None):
        if dtype not in {"float32", "float16", "int8"}:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.dtype = dtype
        self.use_bias = bias is not None
        self.quant_params = quant_params or {}
        self.weights = _copy_parameter(weights, dtype)
        bias_dtype = np.float16 if dtype == "float16" else np.float32
        self.bias = _copy_parameter(bias, bias_dtype) if bias is not None else None
        self.activation = activation

    def forward(self, x):
        xp = get_array_module(x)
        x = x.astype(xp.float32, copy=False)
        if self.dtype == "int8":
            params = self.quant_params["weights"]
            width = self.weights.shape[1]
            out = xp.empty(x.shape[:-1] + (width,), dtype=xp.float32)
            for start in range(0, width, 32):
                end = min(start + 32, width)
                scale = (
                    params.scale[..., start:end]
                    if hasattr(params.scale, "shape")
                    else params.scale
                )
                zero = (
                    params.zero_point[..., start:end]
                    if hasattr(params.zero_point, "shape")
                    else params.zero_point
                )
                if hasattr(zero, "astype"):
                    zero = zero.astype(xp.float32, copy=False)
                block = (self.weights[:, start:end].astype(xp.float32) - zero) * scale
                out[..., start:end] = x @ block
        elif self.dtype == "float16":
            # Accumulate in float32 to avoid half-precision overflow. Blocking
            # also avoids an expanded full copy of the weight matrix.
            width = self.weights.shape[1]
            out = xp.empty(x.shape[:-1] + (width,), dtype=xp.float32)
            for start in range(0, width, 32):
                end = min(start + 32, width)
                out[..., start:end] = x @ self.weights[:, start:end].astype(xp.float32)
        else:
            out = x @ self.weights
        if self.bias is not None:
            out += self.bias
        return self.activation.forward(out) if self.activation is not None else out


class QuantizedActivation(QuantizedLayer):
    def __init__(self, activation_type, alpha=None):
        if activation_type not in {
            "relu",
            "sigmoid",
            "softmax",
            "tanh",
            "leaky_relu",
            "elu",
            "linear",
        }:
            raise ValueError(f"Unsupported activation: {activation_type}")
        self.activation_type = activation_type
        self.alpha = alpha

    def forward(self, x):
        xp = get_array_module(x)
        if self.activation_type == "relu":
            return xp.maximum(0, x)
        if self.activation_type == "sigmoid":
            z = xp.exp(-xp.abs(x))
            return xp.where(x >= 0, 1 / (1 + z), z / (1 + z))
        if self.activation_type == "softmax":
            exp_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
            return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)
        if self.activation_type == "tanh":
            return xp.tanh(x)
        if self.activation_type == "leaky_relu":
            return xp.where(x > 0, x, (0.01 if self.alpha is None else self.alpha) * x)
        if self.activation_type == "elu":
            return xp.where(
                x > 0,
                x,
                (1.0 if self.alpha is None else self.alpha)
                * xp.expm1(xp.minimum(x, 0)),
            )
        return x


class QuantizedDropout(QuantizedLayer):
    """Identity retained for backward-compatible unoptimized exports."""

    def forward(self, x):
        return x


class QuantizedFlatten(QuantizedLayer):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class QuantizedBatchNorm(QuantizedLayer):
    def __init__(self, gamma, beta, running_mean, running_var, eps, dtype):
        self.eps = eps
        self.dtype = dtype
        storage = np.float16 if dtype == "float16" else np.float32
        for name, value in (
            ("gamma", gamma),
            ("beta", beta),
            ("running_mean", running_mean),
            ("running_var", running_var),
        ):
            setattr(self, name, _copy_parameter(value, storage))

    def forward(self, x):
        xp = get_array_module(x)
        # Running statistics need float32 arithmetic even for FP16 storage.
        return (
            self.gamma.astype(xp.float32)
            * (x - self.running_mean.astype(xp.float32))
            / xp.sqrt(self.running_var.astype(xp.float32) + self.eps)
            + self.beta
        )


def _activation_for_inference(activation):
    if activation is None:
        return None
    from neura_command.inference.runtime import _serialize_activation_function

    config = _serialize_activation_function(activation)
    if config["name"] == "linear":
        return None
    return QuantizedActivation(config["name"], config.get("alpha"))


def quantize_model(
    model, config, calibration_data=None, *, optimize=False, prune_threshold=0.0
):
    """Copy a training model into prediction-only primitives on CPU.

    INT8 is weight-only: calibration data is unnecessary and is accepted for
    compatibility. Set optimize=True to fold normalization, remove identities,
    and compact unused hidden channels. Nonzero pruning is lossy and opt-in.
    """
    from neura_command.layers import (
        Dense,
        Dropout,
        BatchNorm,
        Activation,
        Flatten,
        Input,
    )

    layers = []
    for layer in model.layers:
        if isinstance(layer, Dense):
            layers.append(
                QuantizedDense(
                    layer.weights,
                    layer.bias if layer.use_bias else None,
                    "float32",
                    activation=_activation_for_inference(layer.activation),
                )
            )
        elif isinstance(layer, BatchNorm):
            layers.append(
                QuantizedBatchNorm(
                    layer.gamma,
                    layer.beta,
                    layer.running_mean,
                    layer.running_var,
                    layer.epsilon,
                    "float32",
                )
            )
        elif isinstance(layer, Activation):
            layers.append(
                _activation_for_inference(layer.activation) or QuantizedDropout()
            )
        elif isinstance(layer, Flatten):
            layers.append(QuantizedFlatten())
        elif isinstance(layer, (Input, Dropout)):
            layers.append(QuantizedDropout())
        else:
            raise ValueError(f"Cannot export layer type: {type(layer).__name__}")
    if optimize:
        from neura_command.inference.optimization import optimize_layers

        layers = optimize_layers(layers, prune_threshold=prune_threshold)
    elif prune_threshold != 0:
        raise ValueError("Pruning requires optimize=True")

    for i, layer in enumerate(layers):
        if isinstance(layer, QuantizedDense):
            if config.dtype == "int8":
                params = _compute_int8_params(
                    layer.weights, config.symmetric, config.per_channel
                )
                weights = _quantize_to_int8(layer.weights, params)
                layers[i] = QuantizedDense(
                    weights, layer.bias, "int8", {"weights": params}, layer.activation
                )
            else:
                layers[i] = QuantizedDense(
                    layer.weights, layer.bias, config.dtype, activation=layer.activation
                )
        elif isinstance(layer, QuantizedBatchNorm):
            layers[i] = QuantizedBatchNorm(
                layer.gamma,
                layer.beta,
                layer.running_mean,
                layer.running_var,
                layer.eps,
                config.dtype,
            )
    return layers


def get_quantization_stats(model, quantized_layers):
    original_bytes = sum(
        value.nbytes
        for layer in model.layers
        for name, value in vars(layer).items()
        if name in {"weights", "bias", "gamma", "beta", "running_mean", "running_var"}
        and value is not None
    )
    quantized_bytes = sum(layer.memory_bytes for layer in quantized_layers)
    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": original_bytes / quantized_bytes
        if quantized_bytes
        else 1.0,
        "memory_saved_percent": (1 - quantized_bytes / original_bytes) * 100
        if original_bytes
        else 0,
    }
