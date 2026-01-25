"""
Lightweight inference runtime for NeuraCommand.

Provides optimized, inference-only model loading and prediction.
Strips training code for minimal memory footprint on edge devices.
"""

from neura_command.inference.runtime import InferenceModel
from neura_command.inference.quantization import quantize_model, QuantizationConfig

__all__ = ['InferenceModel', 'quantize_model', 'QuantizationConfig']
