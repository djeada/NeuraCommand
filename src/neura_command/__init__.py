"""
NeuraCommand - Lightweight Neural Network Library for Edge Devices

A simple, dependency-light neural network library built on NumPy.
Designed for training and deploying small models on edge devices
where TensorFlow/PyTorch are too heavy.

Example:
    from neura_command import Sequential, Dense, ReLU, Softmax, Adam, CrossEntropyLoss
    
    model = Sequential(
        layers=[
            Dense(784, 128),
            ReLU(),
            Dense(128, 10),
            Softmax(),
        ],
        optimizer=Adam(learning_rate=0.001),
        loss=CrossEntropyLoss(),
    )
    
    model.train(X_train, y_train, epochs=10, batch_size=32)
    predictions = model.predict(X_test)
"""

__version__ = "0.4.0"

# New clean API (recommended)
from neura_command.layers import (
    Layer,
    Dense,
    Dropout,
    BatchNorm,
    Flatten,
    Input,
    Activation,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU,
    ELU,
)

from neura_command.network.sequential import Sequential

from neura_command.network_utils.optimizers import (
    Optimizer,
    SGD,
    Adam,
    RMSprop,
    AdaGrad,
    Adadelta,
    Nadam,
)

from neura_command.network_utils.loss_functions import (
    LossFunction,
    MSELoss,
    CrossEntropyLoss,
    HuberLoss,
)

from neura_command.network_utils.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateScheduler,
    HistoryCallback,
)

from neura_command.serialization import (
    save_model,
    load_model,
    get_model_info,
)

from neura_command.inference import (
    InferenceModel,
    quantize_model,
    QuantizationConfig,
)

from neura_command.backend import (
    get_backend,
    set_default_device,
    get_default_device,
    is_cupy_available,
    to_device,
    to_numpy,
)

__all__ = [
    # Version
    "__version__",
    # Layers
    "Layer",
    "Dense",
    "Dropout",
    "BatchNorm",
    "Flatten",
    "Input",
    "Activation",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LeakyReLU",
    "ELU",
    # Network
    "Sequential",
    # Optimizers
    "Optimizer",
    "SGD",
    "Adam",
    "RMSprop",
    "AdaGrad",
    "Adadelta",
    "Nadam",
    # Losses
    "LossFunction",
    "MSELoss",
    "CrossEntropyLoss",
    "HuberLoss",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "LearningRateScheduler",
    "HistoryCallback",
    # Serialization
    "save_model",
    "load_model",
    "get_model_info",
    # Inference
    "InferenceModel",
    "quantize_model",
    "QuantizationConfig",
    # Backend/Device
    "get_backend",
    "set_default_device",
    "get_default_device",
    "is_cupy_available",
    "to_device",
    "to_numpy",
]
