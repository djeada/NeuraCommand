"""
Lightweight inference runtime for NeuraCommand.

Provides minimal-footprint model execution for edge devices.
No backward pass, no optimizer, no training code.

Usage:
    from neura_command.inference import InferenceModel
    from neura_command import Sequential

    # From saved model
    inference = InferenceModel.from_file('model')
    predictions = inference.predict(X)

    # From Sequential model
    inference = InferenceModel.from_sequential(model)
    predictions = inference.predict(X)

    # With quantization
    inference = InferenceModel.from_sequential(model, quantize='float16')
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from neura_command.backend import to_numpy, to_device, normalize_device, get_backend

_INFERENCE_ARTIFACT_TYPE = "inference_model"
_INFERENCE_FORMAT_VERSION = "2.0"


class InferenceModel:
    """Lightweight inference-only model.

    Optimized for:
    - Minimal memory footprint
    - Fast inference
    - Edge device deployment

    No training capabilities - pure prediction only.
    """

    def __init__(
        self, layers: List, metadata: Optional[Dict[str, Any]] = None, device="cpu"
    ):
        """Create inference model from layers.

        Args:
            layers: List of inference layers (QuantizedLayer or similar)
            metadata: Optional model metadata
        """
        self.layers = layers
        self.metadata = metadata or {}
        self.to(device)

    @property
    def device(self):
        return self._device

    def to(self, device):
        device = normalize_device(device)
        get_backend(device)
        for layer in self.layers:
            layer.to(device)
        self._device = device
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            x: Input data (batch_size, input_features)

        Returns:
            Model predictions
        """
        out = to_device(x, self._device)
        xp = get_backend(self._device)
        out = out.astype(xp.float32, copy=False)
        for layer in self.layers:
            out = layer.forward(out)
        return to_numpy(out)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for predict()."""
        return self.predict(x)

    @classmethod
    def from_sequential(
        cls,
        model,
        quantize: Optional[Literal["float16", "int8"]] = None,
        *,
        device="cpu",
        optimize=False,
        per_channel=False,
        prune_threshold=0.0,
    ) -> "InferenceModel":
        """Create InferenceModel from a Sequential model.

        Args:
            model: Sequential model
            quantize: Optional quantization ('float16' or 'int8')
            device: Inference device; defaults to CPU independently of the source
            optimize: Fold BatchNorm and compact unused hidden channels
            per_channel: Use one INT8 scale per Dense output channel
            prune_threshold: Optional lossy outgoing-weight threshold (requires optimize)

        Returns:
            InferenceModel for inference
        """
        from neura_command.inference.quantization import (
            quantize_model,
            QuantizationConfig,
        )

        layers = quantize_model(
            model,
            QuantizationConfig(dtype=quantize or "float32", per_channel=per_channel),
            optimize=optimize,
            prune_threshold=prune_threshold,
        )

        metadata = {
            "source": "Sequential",
            "num_parameters": model.num_parameters,
            "quantized": quantize or "none",
            "optimized": optimize,
            "prune_threshold": prune_threshold,
            "source_layers": len(model.layers),
        }

        return cls(layers, metadata, device=device)

    @classmethod
    def from_file(
        cls,
        path: str,
        quantize: Optional[Literal["float16", "int8"]] = None,
        *,
        device="cpu",
        **kwargs,
    ) -> "InferenceModel":
        """Load InferenceModel from saved model files.

        Args:
            path: Base path to model files (without extension)
            quantize: Optional quantization to apply when loading

        Returns:
            InferenceModel for inference
        """
        if _is_inference_artifact(path):
            if quantize is not None:
                raise ValueError(
                    "Cannot apply quantization while loading an exported inference artifact. "
                    "Re-export the model with the desired quantization."
                )
            if kwargs:
                raise ValueError(
                    "Export options cannot be applied to an existing inference artifact"
                )
            return cls.load(path, device=device)

        from neura_command.serialization import load_model

        model = load_model(path)
        return cls.from_sequential(model, quantize=quantize, device=device, **kwargs)

    def save(self, path: str) -> None:
        """Save an inference artifact as JSON metadata plus NPZ arrays."""
        _, json_path, npz_path = _artifact_paths(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        arrays: Dict[str, np.ndarray] = {}
        config = {
            "artifact_type": _INFERENCE_ARTIFACT_TYPE,
            "format_version": _INFERENCE_FORMAT_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "layers": [
                _serialize_inference_layer(layer, index, arrays)
                for index, layer in enumerate(self.layers)
            ],
        }

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        np.savez(npz_path, **arrays)

    @classmethod
    def load(cls, path: str, *, device="cpu") -> "InferenceModel":
        """Load an exported inference artifact."""
        original_path, json_path, npz_path = _artifact_paths(path)
        _raise_if_pickle_artifact(original_path)
        _raise_if_pickle_artifact(original_path.with_suffix(".pkl"))
        _raise_if_pickle_artifact(original_path.with_suffix(".pickle"))

        if not json_path.exists():
            raise FileNotFoundError(f"Inference config not found: {json_path}")
        if not npz_path.exists():
            raise FileNotFoundError(f"Inference weights not found: {npz_path}")

        with open(json_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        if config.get("artifact_type") != _INFERENCE_ARTIFACT_TYPE:
            raise ValueError(
                f"{json_path} is not an exported NeuraCommand inference artifact."
            )

        format_version = config.get("format_version", "1.0")
        if str(format_version).split(".", 1)[0] not in {"1", "2"}:
            raise ValueError(
                f"Unsupported inference artifact format version: {format_version}"
            )

        with np.load(npz_path, allow_pickle=False) as weights:
            layers = [
                _deserialize_inference_layer(layer_config, weights)
                for layer_config in config["layers"]
            ]
        return cls(layers=layers, metadata=config.get("metadata"), device=device)

    @staticmethod
    def artifact_paths(path: str) -> Tuple[Path, Path]:
        """Return the JSON and NPZ paths used for an inference artifact."""
        _, json_path, npz_path = _artifact_paths(path)
        return json_path, npz_path

    @property
    def memory_bytes(self) -> int:
        """Parameter storage including quantization scales (excludes Python overhead)."""
        total = 0
        for layer in self.layers:
            if hasattr(layer, "memory_bytes"):
                total += layer.memory_bytes
                continue
            # Handle wrapped layers
            actual_layer = (
                layer.layer if isinstance(layer, _InferenceLayerWrapper) else layer
            )

            if hasattr(actual_layer, "weights"):
                total += actual_layer.weights.nbytes
            if hasattr(actual_layer, "bias") and actual_layer.bias is not None:
                total += actual_layer.bias.nbytes
            if hasattr(actual_layer, "gamma"):
                total += actual_layer.gamma.nbytes
            if hasattr(actual_layer, "beta"):
                total += actual_layer.beta.nbytes
            if hasattr(actual_layer, "running_mean"):
                total += actual_layer.running_mean.nbytes
            if hasattr(actual_layer, "running_var"):
                total += actual_layer.running_var.nbytes
        return total

    def export_c(
        self, path, *, name="neura_model", max_weight_bytes=None, max_scratch_bytes=None
    ):
        """Write C99 source/header and an exact parameter/scratch storage report."""
        from neura_command.inference.c_export import export_c

        return export_c(
            self,
            path,
            name=name,
            max_weight_bytes=max_weight_bytes,
            max_scratch_bytes=max_scratch_bytes,
        )

    def info(self) -> str:
        """Get model info as a formatted string."""
        lines = [
            "InferenceModel",
            f"  Layers: {len(self.layers)}",
            f"  Memory: {self.memory_bytes / 1024:.1f} KB",
            f"  Quantized: {self.metadata.get('quantized', 'none')}",
            f"  Device: {self.device}",
        ]
        return "\n".join(lines)


class _InferenceLayerWrapper:
    """Wrapper for original layers in inference mode."""

    def __init__(self, layer):
        self.layer = layer
        # Ensure eval mode
        if hasattr(layer, "eval_mode"):
            layer.eval_mode()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.layer.forward(x, training=False)

    def to(self, device):
        self.layer.to(device)
        return self


def _wrap_layers_for_inference(layers) -> List:
    """Wrap Sequential layers for inference (no quantization)."""
    return [_InferenceLayerWrapper(layer) for layer in layers]


def _artifact_paths(path: str) -> Tuple[Path, Path, Path]:
    path_obj = Path(path)
    if path_obj.suffix in {".json", ".npz"}:
        base_path = path_obj.with_suffix("")
    else:
        base_path = path_obj
    return (
        base_path,
        Path(f"{base_path}.json"),
        Path(f"{base_path}.npz"),
    )


def _is_inference_artifact(path: str) -> bool:
    _, json_path, _ = _artifact_paths(path)
    if not json_path.exists():
        return False

    with open(json_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config.get("artifact_type") == _INFERENCE_ARTIFACT_TYPE


def _raise_if_pickle_artifact(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    with open(path, "rb") as file:
        header = file.read(2)

    if header[:1] == b"\x80":
        raise ValueError(
            f"Refusing to load pickle artifact '{path}'. "
            "Re-export the model using the safe JSON+NPZ inference format."
        )


def _serialize_activation_function(activation) -> Dict[str, Any]:
    from neura_command.network_utils.activation_functions import (
        ELUActivation,
        LeakyReLUActivation,
    )

    if activation is None:
        return {"name": "linear"}

    activation_map = {
        "ReLUActivation": {"name": "relu"},
        "SigmoidActivation": {"name": "sigmoid"},
        "TanhActivation": {"name": "tanh"},
        "SoftmaxActivation": {"name": "softmax"},
        "LinearActivation": {"name": "linear"},
    }

    if isinstance(activation, LeakyReLUActivation):
        return {"name": "leaky_relu", "alpha": activation.alpha}
    if isinstance(activation, ELUActivation):
        return {"name": "elu", "alpha": activation.alpha}

    activation_name = activation.__class__.__name__
    if activation_name not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_name}")
    return activation_map[activation_name]


def _deserialize_activation_function(config: Optional[Dict[str, Any]]):
    from neura_command.network_utils.activation_functions import (
        ELUActivation,
        LeakyReLUActivation,
        LinearActivation,
        ReLUActivation,
        SigmoidActivation,
        SoftmaxActivation,
        TanhActivation,
    )

    if config is None:
        return None

    activation_name = config.get("name", "linear")
    activation_map = {
        "relu": ReLUActivation,
        "sigmoid": SigmoidActivation,
        "tanh": TanhActivation,
        "softmax": SoftmaxActivation,
        "linear": LinearActivation,
        "leaky_relu": LeakyReLUActivation,
        "elu": ELUActivation,
    }

    activation_class = activation_map[activation_name]
    if activation_name in {"leaky_relu", "elu"}:
        return activation_class(alpha=config.get("alpha"))
    return activation_class()


def _serialize_inference_layer(
    layer, layer_index: int, arrays: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    from neura_command.inference.quantization import (
        QuantizedActivation,
        QuantizedBatchNorm,
        QuantizedDense,
        QuantizedDropout,
        QuantizedFlatten,
    )

    if isinstance(layer, _InferenceLayerWrapper):
        return _serialize_wrapped_layer(layer.layer, layer_index, arrays)
    if isinstance(layer, QuantizedDense):
        return _serialize_quantized_dense(layer, layer_index, arrays)
    if isinstance(layer, QuantizedBatchNorm):
        return _serialize_quantized_batchnorm(layer, layer_index, arrays)
    if isinstance(layer, QuantizedActivation):
        return {
            "runtime": "quantized",
            "class": "QuantizedActivation",
            "activation_type": layer.activation_type,
            "alpha": layer.alpha,
        }
    if isinstance(layer, QuantizedDropout):
        return {
            "runtime": "quantized",
            "class": "QuantizedDropout",
        }
    if isinstance(layer, QuantizedFlatten):
        return {"runtime": "quantized", "class": "QuantizedFlatten"}

    raise TypeError(f"Unsupported inference layer type: {type(layer).__name__}")


def _serialize_wrapped_layer(
    layer, layer_index: int, arrays: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    from neura_command.layers import (
        Activation,
        BatchNorm,
        Dense,
        Dropout,
        ELU,
        Flatten,
        Input,
        LeakyReLU,
        ReLU,
        Sigmoid,
        Softmax,
        Tanh,
    )

    if isinstance(layer, Dense):
        weights_key = f"layer_{layer_index}_weights"
        arrays[weights_key] = to_numpy(layer.weights)
        config = {
            "runtime": "wrapped",
            "class": "Dense",
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "use_bias": layer.use_bias,
            "weights_key": weights_key,
            "activation": _serialize_activation_function(layer.activation),
        }
        if layer.use_bias and layer.bias is not None:
            bias_key = f"layer_{layer_index}_bias"
            arrays[bias_key] = to_numpy(layer.bias)
            config["bias_key"] = bias_key
        return config

    if isinstance(layer, BatchNorm):
        gamma_key = f"layer_{layer_index}_gamma"
        beta_key = f"layer_{layer_index}_beta"
        mean_key = f"layer_{layer_index}_running_mean"
        var_key = f"layer_{layer_index}_running_var"
        arrays[gamma_key] = to_numpy(layer.gamma)
        arrays[beta_key] = to_numpy(layer.beta)
        arrays[mean_key] = to_numpy(layer.running_mean)
        arrays[var_key] = to_numpy(layer.running_var)
        return {
            "runtime": "wrapped",
            "class": "BatchNorm",
            "num_features": layer.num_features,
            "epsilon": layer.epsilon,
            "momentum": layer.momentum,
            "gamma_key": gamma_key,
            "beta_key": beta_key,
            "running_mean_key": mean_key,
            "running_var_key": var_key,
        }

    if isinstance(layer, Dropout):
        return {
            "runtime": "wrapped",
            "class": "Dropout",
            "rate": layer.rate,
        }

    if isinstance(layer, Flatten):
        return {"runtime": "wrapped", "class": "Flatten"}

    if isinstance(layer, Input):
        return {
            "runtime": "wrapped",
            "class": "Input",
            "shape": list(layer.shape),
        }

    if isinstance(layer, ReLU):
        return {"runtime": "wrapped", "class": "ReLU"}
    if isinstance(layer, Sigmoid):
        return {"runtime": "wrapped", "class": "Sigmoid"}
    if isinstance(layer, Softmax):
        return {"runtime": "wrapped", "class": "Softmax"}
    if isinstance(layer, Tanh):
        return {"runtime": "wrapped", "class": "Tanh"}
    if isinstance(layer, LeakyReLU):
        return {
            "runtime": "wrapped",
            "class": "LeakyReLU",
            "alpha": layer.activation.alpha,
        }
    if isinstance(layer, ELU):
        return {
            "runtime": "wrapped",
            "class": "ELU",
            "alpha": layer.activation.alpha,
        }
    if isinstance(layer, Activation):
        return {
            "runtime": "wrapped",
            "class": "Activation",
            "activation": _serialize_activation_function(layer.activation),
        }

    raise TypeError(f"Unsupported wrapped inference layer type: {type(layer).__name__}")


def _serialize_quantized_dense(
    layer, layer_index: int, arrays: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    weights_key = f"layer_{layer_index}_weights"
    arrays[weights_key] = to_numpy(layer.weights)

    config = {
        "runtime": "quantized",
        "class": "QuantizedDense",
        "dtype": layer.dtype,
        "use_bias": layer.use_bias,
        "weights_key": weights_key,
        "activation": (
            {"name": layer.activation.activation_type, "alpha": layer.activation.alpha}
            if layer.activation is not None
            else None
        ),
    }

    if layer.use_bias and layer.bias is not None:
        bias_key = f"layer_{layer_index}_bias"
        arrays[bias_key] = to_numpy(layer.bias)
        config["bias_key"] = bias_key

    if layer.dtype == "int8":
        params = layer.quant_params["weights"]
        config["quant_params"] = {
            "weights": {
                "scale": to_numpy(params.scale).tolist(),
                "zero_point": to_numpy(params.zero_point).tolist(),
                "min_val": params.min_val,
                "max_val": params.max_val,
            }
        }

    return config


def _serialize_quantized_batchnorm(
    layer, layer_index: int, arrays: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    gamma_key = f"layer_{layer_index}_gamma"
    beta_key = f"layer_{layer_index}_beta"
    mean_key = f"layer_{layer_index}_running_mean"
    var_key = f"layer_{layer_index}_running_var"
    arrays[gamma_key] = to_numpy(layer.gamma)
    arrays[beta_key] = to_numpy(layer.beta)
    arrays[mean_key] = to_numpy(layer.running_mean)
    arrays[var_key] = to_numpy(layer.running_var)

    return {
        "runtime": "quantized",
        "class": "QuantizedBatchNorm",
        "dtype": layer.dtype,
        "eps": layer.eps,
        "gamma_key": gamma_key,
        "beta_key": beta_key,
        "running_mean_key": mean_key,
        "running_var_key": var_key,
    }


def _deserialize_inference_layer(config: Dict[str, Any], arrays) -> Any:
    runtime = config["runtime"]
    if runtime == "wrapped":
        return _InferenceLayerWrapper(_deserialize_wrapped_layer(config, arrays))
    if runtime == "quantized":
        return _deserialize_quantized_layer(config, arrays)
    raise ValueError(f"Unknown inference runtime type: {runtime}")


def _deserialize_wrapped_layer(config: Dict[str, Any], arrays):
    from neura_command.layers import (
        Activation,
        BatchNorm,
        Dense,
        Dropout,
        ELU,
        Flatten,
        Input,
        LeakyReLU,
        ReLU,
        Sigmoid,
        Softmax,
        Tanh,
    )

    class_name = config["class"]

    if class_name == "Dense":
        layer = Dense(
            in_features=config["in_features"],
            out_features=config["out_features"],
            activation=_deserialize_activation_function(config.get("activation")),
            use_bias=config.get("use_bias", True),
            device="cpu",
        )
        layer.weights[...] = arrays[config["weights_key"]]
        if layer.use_bias and "bias_key" in config:
            layer.bias[...] = arrays[config["bias_key"]]
        return layer

    if class_name == "BatchNorm":
        layer = BatchNorm(
            num_features=config["num_features"],
            epsilon=config.get("epsilon", 1e-5),
            momentum=config.get("momentum", 0.1),
            device="cpu",
        )
        layer.gamma[...] = arrays[config["gamma_key"]]
        layer.beta[...] = arrays[config["beta_key"]]
        layer.running_mean = arrays[config["running_mean_key"]]
        layer.running_var = arrays[config["running_var_key"]]
        return layer

    if class_name == "Dropout":
        return Dropout(rate=config["rate"], device="cpu")
    if class_name == "Flatten":
        return Flatten()
    if class_name == "Input":
        return Input(shape=tuple(config["shape"]))
    if class_name == "ReLU":
        return ReLU(device="cpu")
    if class_name == "Sigmoid":
        return Sigmoid(device="cpu")
    if class_name == "Softmax":
        return Softmax(device="cpu")
    if class_name == "Tanh":
        return Tanh(device="cpu")
    if class_name == "LeakyReLU":
        return LeakyReLU(alpha=config.get("alpha", 0.01), device="cpu")
    if class_name == "ELU":
        return ELU(alpha=config.get("alpha", 1.0), device="cpu")
    if class_name == "Activation":
        return Activation(
            _deserialize_activation_function(config["activation"]), device="cpu"
        )

    raise ValueError(f"Unsupported wrapped inference layer type: {class_name}")


def _deserialize_quantized_layer(config: Dict[str, Any], arrays):
    from neura_command.inference.quantization import (
        Int8QuantizationParams,
        QuantizedActivation,
        QuantizedBatchNorm,
        QuantizedDense,
        QuantizedDropout,
        QuantizedFlatten,
    )

    class_name = config["class"]

    if class_name == "QuantizedDense":
        bias = arrays[config["bias_key"]] if "bias_key" in config else None
        quant_params = None
        if config["dtype"] == "int8":
            params = config["quant_params"]["weights"]
            quant_params = {
                "weights": Int8QuantizationParams(
                    scale=(
                        np.asarray(params["scale"], dtype=np.float32)
                        if isinstance(params["scale"], list)
                        else params["scale"]
                    ),
                    zero_point=(
                        np.asarray(params["zero_point"], dtype=np.int32)
                        if isinstance(params["zero_point"], list)
                        else params["zero_point"]
                    ),
                    min_val=params["min_val"],
                    max_val=params["max_val"],
                )
            }
        return QuantizedDense(
            weights=arrays[config["weights_key"]],
            bias=bias,
            dtype=config["dtype"],
            quant_params=quant_params,
            activation=(
                QuantizedActivation(
                    config["activation"]["name"], config["activation"].get("alpha")
                )
                if config.get("activation")
                else None
            ),
        )

    if class_name == "QuantizedBatchNorm":
        return QuantizedBatchNorm(
            gamma=arrays[config["gamma_key"]],
            beta=arrays[config["beta_key"]],
            running_mean=arrays[config["running_mean_key"]],
            running_var=arrays[config["running_var_key"]],
            eps=config["eps"],
            dtype=config["dtype"],
        )

    if class_name == "QuantizedActivation":
        return QuantizedActivation(
            activation_type=config["activation_type"],
            alpha=config.get("alpha"),
        )

    if class_name == "QuantizedDropout":
        return QuantizedDropout()
    if class_name == "QuantizedFlatten":
        return QuantizedFlatten()

    raise ValueError(f"Unsupported quantized inference layer type: {class_name}")
