"""
Deprecated modules.

These modules are kept for backward compatibility but should not be used
in new code. Use the new API instead:

Old:
    from neura_command.neuron.perceptron.perceptron import Perceptron
    from neura_command.layer.layer import Layer
    from neura_command.network.feed_forward_network import FeedForwardNetwork

New:
    from neura_command import Sequential, Dense, ReLU, Adam

    model = Sequential([
        Dense(input_size, hidden_size),
        ReLU(),
        Dense(hidden_size, output_size),
    ], optimizer=Adam(), loss=MSELoss())
"""

import warnings

_WARNED = False


def warn_deprecated() -> None:
    """Emit a single deprecation warning when legacy code is actually used."""
    global _WARNED
    if _WARNED:
        return

    warnings.warn(
        "The neura_command.deprecated module contains legacy code. "
        "Please migrate to the new API (Sequential, Dense, etc.)",
        DeprecationWarning,
        stacklevel=2,
    )
    _WARNED = True
