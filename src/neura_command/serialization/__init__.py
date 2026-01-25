"""
Model serialization for NeuraCommand.

Provides safe, portable model saving and loading using JSON for architecture
and NPZ for weights. No pickle - no security risks.

Format:
    model.json  - Architecture, hyperparameters, metadata
    weights.npz - NumPy arrays for all layer parameters
    
Example:
    from neura_command import Sequential, Dense, ReLU
    from neura_command.serialization import save_model, load_model
    
    model = Sequential([Dense(10, 5), ReLU()])
    save_model(model, 'my_model')
    
    loaded = load_model('my_model')
"""

from neura_command.serialization.model_io import save_model, load_model, get_model_info

__all__ = ["save_model", "load_model", "get_model_info"]
