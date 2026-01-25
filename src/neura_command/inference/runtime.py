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
import numpy as np
from typing import List, Optional, Literal, Dict, Any
from pathlib import Path


class InferenceModel:
    """Lightweight inference-only model.

    Optimized for:
    - Minimal memory footprint
    - Fast inference
    - Edge device deployment

    No training capabilities - pure prediction only.
    """

    def __init__(self, layers: List, metadata: Optional[Dict[str, Any]] = None):
        """Create inference model from layers.

        Args:
            layers: List of inference layers (QuantizedLayer or similar)
            metadata: Optional model metadata
        """
        self.layers = layers
        self.metadata = metadata or {}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            x: Input data (batch_size, input_features)

        Returns:
            Model predictions
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for predict()."""
        return self.predict(x)

    @classmethod
    def from_sequential(
        cls,
        model,
        quantize: Optional[Literal["float16", "int8"]] = None,
    ) -> "InferenceModel":
        """Create InferenceModel from a Sequential model.

        Args:
            model: Sequential model
            quantize: Optional quantization ('float16' or 'int8')

        Returns:
            InferenceModel for inference
        """
        if quantize:
            from neura_command.inference.quantization import (
                quantize_model,
                QuantizationConfig,
            )

            layers = quantize_model(model, QuantizationConfig(dtype=quantize))
        else:
            # Use original layers in eval mode
            layers = _wrap_layers_for_inference(model.layers)

        metadata = {
            "source": "Sequential",
            "num_parameters": model.num_parameters,
            "quantized": quantize or "none",
        }

        return cls(layers, metadata)

    @classmethod
    def from_file(
        cls,
        path: str,
        quantize: Optional[Literal["float16", "int8"]] = None,
    ) -> "InferenceModel":
        """Load InferenceModel from saved model files.

        Args:
            path: Base path to model files (without extension)
            quantize: Optional quantization to apply when loading

        Returns:
            InferenceModel for inference
        """
        from neura_command.serialization import load_model

        model = load_model(path)
        return cls.from_sequential(model, quantize=quantize)

    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage of the model in bytes."""
        total = 0
        for layer in self.layers:
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

    def info(self) -> str:
        """Get model info as a formatted string."""
        lines = [
            "InferenceModel",
            f"  Layers: {len(self.layers)}",
            f"  Memory: {self.memory_bytes / 1024:.1f} KB",
            f"  Quantized: {self.metadata.get('quantized', 'none')}",
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


def _wrap_layers_for_inference(layers) -> List:
    """Wrap Sequential layers for inference (no quantization)."""
    return [_InferenceLayerWrapper(layer) for layer in layers]
