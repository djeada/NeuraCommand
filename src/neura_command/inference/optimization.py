"""Export-time graph simplification on private float32 inference copies."""

import numpy as np

from neura_command.inference.quantization import (
    QuantizedActivation,
    QuantizedBatchNorm,
    QuantizedDense,
    QuantizedDropout,
)


def optimize_layers(layers, prune_threshold=0.0):
    """Fold linear Dense/BatchNorm pairs and compact hidden channels.

    At threshold zero, remove only channels with zero outgoing weights or
    provably zero output. A positive threshold also removes channels whose
    outgoing weights are all below that threshold; validate accuracy afterward.
    Output dimensions and the source training model are preserved.
    """
    if not np.isfinite(prune_threshold) or prune_threshold < 0:
        raise ValueError("prune_threshold must be finite and non-negative")
    result = []
    for layer in layers:
        if isinstance(layer, QuantizedDropout):
            continue
        previous = result[-1] if result else None
        if (
            isinstance(layer, QuantizedBatchNorm)
            and isinstance(previous, QuantizedDense)
            and previous.activation is None
        ):
            if previous.weights.shape[1] != layer.gamma.size:
                raise ValueError("Dense and BatchNorm feature counts do not match")
            scale = layer.gamma / np.sqrt(layer.running_var + layer.eps)
            previous.weights *= scale
            bias = (
                previous.bias
                if previous.bias is not None
                else np.zeros_like(layer.beta)
            )
            previous.bias = ((bias - layer.running_mean) * scale + layer.beta).astype(
                np.float32
            )
            previous.use_bias = True
            continue
        if (
            isinstance(layer, QuantizedActivation)
            and isinstance(previous, QuantizedDense)
            and previous.activation is None
        ):
            previous.activation = layer
            continue
        result.append(layer)

    # Work backwards so compaction can expose unused channels further upstream.
    for i in reversed(range(len(result))):
        first = result[i]
        if not isinstance(first, QuantizedDense):
            continue
        activations = [first.activation] if first.activation is not None else []
        j = i + 1
        while j < len(result) and isinstance(result[j], QuantizedActivation):
            activations.append(result[j])
            j += 1
        if j == len(result) or not isinstance(result[j], QuantizedDense):
            continue
        # Softmax couples all channels; slicing it would change the function.
        if any(activation.activation_type == "softmax" for activation in activations):
            continue
        second = result[j]
        if first.weights.shape[1] != second.weights.shape[0]:
            raise ValueError("Adjacent Dense feature counts do not match")
        scores = np.max(np.abs(second.weights), axis=1)
        keep = scores > prune_threshold
        zero_preserving = all(
            activation.activation_type != "sigmoid" for activation in activations
        )
        if zero_preserving:
            inactive = np.all(first.weights == 0, axis=0)
            if first.bias is not None:
                inactive &= first.bias.reshape(-1) == 0
            keep &= ~inactive
        if not np.any(keep):
            keep[int(np.argmax(scores))] = True  # Keep valid nonempty matrix shapes.
        first.weights = first.weights[:, keep].copy()
        if first.bias is not None:
            first.bias = first.bias[:, keep].copy()
        second.weights = second.weights[keep, :].copy()
    for layer in result:
        if (
            isinstance(layer, QuantizedDense)
            and layer.bias is not None
            and np.all(layer.bias == 0)
        ):
            layer.bias = None
            layer.use_bias = False
    return result
