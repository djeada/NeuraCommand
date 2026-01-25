"""
Model quantization for NeuraCommand.

Supports:
- float16: Half precision (50% memory reduction, minimal accuracy loss)
- int8: 8-bit quantization (75% memory reduction, some accuracy trade-off)

Usage:
    from neura_command import Sequential, Dense
    from neura_command.inference import quantize_model, QuantizationConfig
    
    model = Sequential([Dense(784, 128), Dense(128, 10)])
    
    # Quantize to float16
    quantized = quantize_model(model, QuantizationConfig(dtype='float16'))
    
    # Quantize to int8 with calibration data
    quantized = quantize_model(model, QuantizationConfig(dtype='int8'), calibration_data=X_sample)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        dtype: Target data type ('float16' or 'int8')
        symmetric: Use symmetric quantization for int8 (default: True)
        per_channel: Quantize per output channel for better accuracy (default: False)
    """

    dtype: Literal["float16", "int8"] = "float16"
    symmetric: bool = True
    per_channel: bool = False


@dataclass
class Int8QuantizationParams:
    """Parameters for int8 quantization of a single array."""

    scale: float
    zero_point: int
    min_val: float
    max_val: float


def _compute_int8_params(
    weights: np.ndarray, symmetric: bool = True
) -> Int8QuantizationParams:
    """Compute quantization parameters for int8 conversion.

    Args:
        weights: Array to quantize
        symmetric: If True, zero_point is always 0 (simpler but less accurate)

    Returns:
        Quantization parameters
    """
    min_val = float(weights.min())
    max_val = float(weights.max())

    if symmetric:
        abs_max = max(abs(min_val), abs(max_val))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        zero_point = 0
    else:
        scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
        zero_point = int(round(-min_val / scale)) if scale > 0 else 0
        zero_point = max(-128, min(127, zero_point))

    return Int8QuantizationParams(
        scale=scale, zero_point=zero_point, min_val=min_val, max_val=max_val
    )


def _quantize_to_int8(
    weights: np.ndarray, params: Int8QuantizationParams
) -> np.ndarray:
    """Quantize weights to int8."""
    quantized = np.round(weights / params.scale) + params.zero_point
    return np.clip(quantized, -128, 127).astype(np.int8)


def _dequantize_from_int8(
    quantized: np.ndarray, params: Int8QuantizationParams
) -> np.ndarray:
    """Dequantize int8 weights back to float32."""
    return (quantized.astype(np.float32) - params.zero_point) * params.scale


class QuantizedLayer:
    """Base class for quantized layers."""

    pass


class QuantizedDense(QuantizedLayer):
    """Quantized Dense layer for inference.

    Stores weights in quantized format and dequantizes during forward pass.
    For int8, this provides ~75% memory reduction.
    """

    def __init__(
        self,
        weights: np.ndarray,
        bias: Optional[np.ndarray],
        dtype: str,
        quant_params: Optional[Dict[str, Int8QuantizationParams]] = None,
    ):
        self.dtype = dtype
        self.use_bias = bias is not None
        self.quant_params = quant_params or {}

        if dtype == "float16":
            self.weights = weights.astype(np.float16)
            self.bias = bias.astype(np.float16) if bias is not None else None
        elif dtype == "int8":
            self.weights = weights  # Already int8
            self.bias = bias  # Bias usually kept at higher precision
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with quantized weights."""
        if self.dtype == "float16":
            # Convert input to float16, compute, convert back
            x_fp16 = x.astype(np.float16)
            out = x_fp16 @ self.weights
            if self.bias is not None:
                out += self.bias
            return out.astype(np.float32)
        elif self.dtype == "int8":
            # Dequantize weights for computation
            w_dequant = _dequantize_from_int8(
                self.weights, self.quant_params["weights"]
            )
            out = x @ w_dequant
            if self.bias is not None:
                out += self.bias
            return out
        return x


class QuantizedActivation(QuantizedLayer):
    """Quantized activation layer (no quantization needed, just the function)."""

    def __init__(self, activation_type: str):
        self.activation_type = activation_type

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.activation_type == "relu":
            return np.maximum(0, x)
        elif self.activation_type == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation_type == "softmax":
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        elif self.activation_type == "tanh":
            return np.tanh(x)
        else:
            return x


class QuantizedDropout(QuantizedLayer):
    """Quantized Dropout - always in eval mode (passthrough)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


class QuantizedBatchNorm(QuantizedLayer):
    """Quantized BatchNorm - uses running stats (eval mode only)."""

    def __init__(
        self,
        gamma: np.ndarray,
        beta: np.ndarray,
        running_mean: np.ndarray,
        running_var: np.ndarray,
        eps: float,
        dtype: str,
    ):
        self.eps = eps
        self.dtype = dtype

        if dtype == "float16":
            self.gamma = gamma.astype(np.float16)
            self.beta = beta.astype(np.float16)
            self.running_mean = running_mean.astype(np.float16)
            self.running_var = running_var.astype(np.float16)
        else:
            self.gamma = gamma
            self.beta = beta
            self.running_mean = running_mean
            self.running_var = running_var

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.dtype == "float16":
            x_fp16 = x.astype(np.float16)
            x_norm = (x_fp16 - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
            return out.astype(np.float32)
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta


def quantize_model(
    model,
    config: QuantizationConfig,
    calibration_data: Optional[np.ndarray] = None,
) -> List[QuantizedLayer]:
    """Quantize a Sequential model for inference.

    Args:
        model: Sequential model to quantize
        config: Quantization configuration
        calibration_data: Sample data for int8 calibration (optional)

    Returns:
        List of quantized layers for InferenceModel

    Example:
        quantized_layers = quantize_model(model, QuantizationConfig(dtype='float16'))
        inference_model = InferenceModel(quantized_layers)
    """
    from neura_command.layers import Dense, Dropout, BatchNorm
    from neura_command.layers.activation import (
        ReLU,
        Sigmoid,
        Softmax,
        Tanh,
        LeakyReLU,
        ELU,
    )

    quantized_layers = []

    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = layer.weights.copy()
            bias = layer.bias.copy() if layer.use_bias else None

            if config.dtype == "float16":
                qlayer = QuantizedDense(weights, bias, dtype="float16")
            elif config.dtype == "int8":
                # Compute quantization params
                w_params = _compute_int8_params(weights, symmetric=config.symmetric)
                q_weights = _quantize_to_int8(weights, w_params)
                # Keep bias in float32 for accuracy
                qlayer = QuantizedDense(
                    q_weights, bias, dtype="int8", quant_params={"weights": w_params}
                )
            quantized_layers.append(qlayer)

        elif isinstance(layer, Dropout):
            quantized_layers.append(QuantizedDropout())

        elif isinstance(layer, BatchNorm):
            qlayer = QuantizedBatchNorm(
                gamma=layer.gamma.copy(),
                beta=layer.beta.copy(),
                running_mean=layer.running_mean.copy(),
                running_var=layer.running_var.copy(),
                eps=layer.epsilon,
                dtype=config.dtype,
            )
            quantized_layers.append(qlayer)

        elif isinstance(layer, ReLU):
            quantized_layers.append(QuantizedActivation("relu"))
        elif isinstance(layer, Sigmoid):
            quantized_layers.append(QuantizedActivation("sigmoid"))
        elif isinstance(layer, Softmax):
            quantized_layers.append(QuantizedActivation("softmax"))
        elif isinstance(layer, Tanh):
            quantized_layers.append(QuantizedActivation("tanh"))
        elif isinstance(layer, (LeakyReLU, ELU)):
            # For LeakyReLU/ELU, fall back to generic handling
            quantized_layers.append(
                QuantizedActivation(layer.__class__.__name__.lower())
            )
        else:
            raise ValueError(f"Cannot quantize layer type: {type(layer).__name__}")

    return quantized_layers


def get_quantization_stats(
    model, quantized_layers: List[QuantizedLayer]
) -> Dict[str, Any]:
    """Get statistics about quantization.

    Args:
        model: Original Sequential model
        quantized_layers: Quantized layers

    Returns:
        Dict with memory usage, compression ratio, etc.
    """
    from neura_command.layers import Dense

    original_bytes = 0
    quantized_bytes = 0

    for orig_layer, q_layer in zip(model.layers, quantized_layers):
        if isinstance(orig_layer, Dense):
            # Original size (float32 = 4 bytes)
            original_bytes += orig_layer.weights.nbytes
            if orig_layer.use_bias:
                original_bytes += orig_layer.bias.nbytes

            # Quantized size
            quantized_bytes += q_layer.weights.nbytes
            if q_layer.bias is not None:
                quantized_bytes += q_layer.bias.nbytes

    return {
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "compression_ratio": (
            original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
        ),
        "memory_saved_percent": (
            (1 - quantized_bytes / original_bytes) * 100 if original_bytes > 0 else 0
        ),
    }
