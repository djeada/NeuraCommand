"""
Model I/O for saving and loading NeuraCommand models.

Uses a safe format:
- model.json: Architecture and hyperparameters (human-readable)
- weights.npz: NumPy arrays (binary, efficient)

No pickle is used - this format is safe to load from untrusted sources.
"""

import json
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from datetime import datetime, timezone

import numpy as np

from neura_command import __version__

if TYPE_CHECKING:
    from neura_command.network.sequential import Sequential


# Registry of layer classes for deserialization
_LAYER_REGISTRY: Dict[str, type] = {}


def register_layer(cls: type) -> type:
    """Register a layer class for serialization."""
    _LAYER_REGISTRY[cls.__name__] = cls
    return cls


def _ensure_registry():
    """Ensure all layer classes are registered."""
    if not _LAYER_REGISTRY:
        from neura_command.layers import (
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

        for cls in [
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
        ]:
            register_layer(cls)


def _serialize_layer(layer) -> Dict[str, Any]:
    """Serialize a layer to a dictionary."""
    from neura_command.layers import (
        Dense,
        Dropout,
        BatchNorm,
        Flatten,
        Input,
        Activation,
    )
    from neura_command.layers.activation import LeakyReLU, ELU

    config = {"class": layer.__class__.__name__}

    if isinstance(layer, Dense):
        config.update(
            {
                "in_features": layer.in_features,
                "out_features": layer.out_features,
                "use_bias": layer.use_bias,
                "activation": (
                    layer.activation.__class__.__name__ if layer.activation else None
                ),
            }
        )
        if layer.activation is not None and hasattr(layer.activation, "alpha"):
            config["activation_alpha"] = layer.activation.alpha
    elif isinstance(layer, Dropout):
        config["rate"] = layer.rate
    elif isinstance(layer, BatchNorm):
        config.update(
            {
                "num_features": layer.num_features,
                "epsilon": layer.epsilon,
                "momentum": layer.momentum,
            }
        )
    elif isinstance(layer, Flatten):
        pass  # No config needed
    elif isinstance(layer, Input):
        config["shape"] = list(layer.shape)
    elif isinstance(layer, (LeakyReLU, ELU)):
        config["alpha"] = layer.activation.alpha
    elif isinstance(layer, Activation):
        config["activation"] = layer.activation.__class__.__name__
        if hasattr(layer.activation, "alpha"):
            config["activation_alpha"] = layer.activation.alpha

    return config


def _deserialize_layer(config: Dict[str, Any]) -> Any:
    """Deserialize a layer from a dictionary."""
    _ensure_registry()

    class_name = config["class"]

    if class_name not in _LAYER_REGISTRY:
        raise ValueError(f"Unknown layer type: {class_name}")

    layer_class = _LAYER_REGISTRY[class_name]

    # Handle different layer types
    if class_name == "Dense":
        from neura_command.network_utils.activation_functions import (
            ReLUActivation,
            SigmoidActivation,
            TanhActivation,
            SoftmaxActivation,
            LeakyReLUActivation,
            ELUActivation,
            LinearActivation,
        )

        activation_map = {
            "ReLUActivation": ReLUActivation(),
            "SigmoidActivation": SigmoidActivation(),
            "TanhActivation": TanhActivation(),
            "SoftmaxActivation": SoftmaxActivation(),
            "LeakyReLUActivation": LeakyReLUActivation(),
            "ELUActivation": ELUActivation(),
            "LinearActivation": LinearActivation(),
            None: None,
        }
        activation_name = config.get("activation")
        if activation_name not in activation_map:
            raise ValueError(f"Unknown Dense activation: {activation_name}")
        activation = activation_map[activation_name]
        if (
            activation_name in {"LeakyReLUActivation", "ELUActivation"}
            and "activation_alpha" in config
        ):
            activation.alpha = config["activation_alpha"]
        return layer_class(
            in_features=config["in_features"],
            out_features=config["out_features"],
            use_bias=config.get("use_bias", True),
            activation=activation,
        )
    elif class_name == "Dropout":
        return layer_class(rate=config["rate"])
    elif class_name == "Input":
        return layer_class(shape=tuple(config["shape"]))
    elif class_name == "BatchNorm":
        return layer_class(
            num_features=config["num_features"],
            epsilon=config.get("epsilon", 1e-5),
            momentum=config.get("momentum", 0.1),
        )
    elif class_name in ("Flatten", "ReLU", "Sigmoid", "Tanh", "Softmax"):
        return layer_class()
    elif class_name == "LeakyReLU":
        return layer_class(alpha=config.get("alpha", 0.01))
    elif class_name == "ELU":
        return layer_class(alpha=config.get("alpha", 1.0))
    elif class_name == "Activation":
        activation_name = config.get("activation", "ReLUActivation")
        # Map activation class names to string names
        name_map = {
            "ReLUActivation": "relu",
            "SigmoidActivation": "sigmoid",
            "TanhActivation": "tanh",
            "SoftmaxActivation": "softmax",
            "LeakyReLUActivation": "leaky_relu",
            "ELUActivation": "elu",
            "LinearActivation": "linear",
        }
        if activation_name not in name_map:
            raise ValueError(f"Unknown Activation layer function: {activation_name}")
        layer = layer_class(name_map[activation_name])
        if hasattr(layer.activation, "alpha") and "activation_alpha" in config:
            layer.activation.alpha = config["activation_alpha"]
        return layer
    else:
        raise ValueError(f"Don't know how to deserialize: {class_name}")


def _filter_constructor_kwargs(factory, config: Dict[str, Any]) -> Dict[str, Any]:
    """Filter config keys to only those accepted by the target constructor."""
    signature = inspect.signature(factory)
    return {key: value for key, value in config.items() if key in signature.parameters}


def _serialize_optimizer(optimizer) -> Dict[str, Any]:
    """Serialize optimizer configuration."""
    from neura_command.network_utils.optimizers import (
        SGD,
        Adam,
        RMSprop,
        AdaGrad,
        Adadelta,
        Nadam,
    )

    config = {"class": optimizer.__class__.__name__}

    if isinstance(optimizer, SGD):
        config.update(
            {
                "learning_rate": optimizer.learning_rate,
                "momentum": optimizer.momentum,
                "nesterov": optimizer.nesterov,
                "weight_decay": optimizer.weight_decay,
                "l1_regularization": optimizer.l1_regularization,
            }
        )
    elif isinstance(optimizer, (Adam, Nadam)):
        config.update(
            {
                "learning_rate": optimizer.learning_rate,
                "beta1": optimizer.beta1,
                "beta2": optimizer.beta2,
                "epsilon": optimizer.epsilon,
                "weight_decay": optimizer.weight_decay,
            }
        )
    elif isinstance(optimizer, (RMSprop, Adadelta)):
        config.update(
            {
                "learning_rate": optimizer.learning_rate,
                "rho": optimizer.rho,
                "epsilon": optimizer.epsilon,
                "weight_decay": optimizer.weight_decay,
            }
        )
    elif isinstance(optimizer, AdaGrad):
        config.update(
            {
                "learning_rate": optimizer.learning_rate,
                "epsilon": optimizer.epsilon,
            }
        )

    return config


def _deserialize_optimizer(config: Dict[str, Any]):
    """Deserialize optimizer from configuration."""
    from neura_command.network_utils.optimizers import (
        SGD,
        Adam,
        RMSprop,
        AdaGrad,
        Adadelta,
        Nadam,
    )

    optimizer_map = {
        "SGD": SGD,
        "Adam": Adam,
        "RMSprop": RMSprop,
        "AdaGrad": AdaGrad,
        "Adadelta": Adadelta,
        "Nadam": Nadam,
    }

    class_name = config.pop("class")
    if class_name not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {class_name}")

    return optimizer_map[class_name](
        **_filter_constructor_kwargs(optimizer_map[class_name], config)
    )


def _serialize_loss(loss) -> Dict[str, Any]:
    """Serialize loss function configuration."""
    from neura_command.network_utils.loss_functions import (
        HuberLoss,
    )

    config = {"class": loss.__class__.__name__}

    if isinstance(loss, HuberLoss):
        config["delta"] = loss.delta

    return config


def _deserialize_loss(config: Dict[str, Any]):
    """Deserialize loss function from configuration."""
    from neura_command.network_utils.loss_functions import (
        MSELoss,
        CrossEntropyLoss,
        HuberLoss,
        DifferenceLoss,
    )

    loss_map = {
        "MSELoss": MSELoss,
        "CrossEntropyLoss": CrossEntropyLoss,
        "HuberLoss": HuberLoss,
        "DifferenceLoss": DifferenceLoss,
    }

    class_name = config.pop("class")
    if class_name not in loss_map:
        raise ValueError(f"Unknown loss function: {class_name}")

    return loss_map[class_name](
        **_filter_constructor_kwargs(loss_map[class_name], config)
    )


def save_model(
    model,
    path: Union[str, Path],
    include_optimizer: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model to disk.

    Creates two files:
    - {path}.json: Model architecture and configuration
    - {path}.npz: Model weights

    Args:
        model: Sequential model to save
        path: Base path for saved files (without extension)
        include_optimizer: Whether to save optimizer state
        metadata: Optional metadata to include

    Example:
        save_model(model, 'models/my_model')
        # Creates: models/my_model.json, models/my_model.npz
    """
    from neura_command.backend import to_numpy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build model configuration
    config = {
        "neura_command_version": __version__,
        "format_version": "1.0",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "class": model.__class__.__name__,
            "layers": [_serialize_layer(layer) for layer in model.layers],
        },
        "loss": _serialize_loss(model.loss),
    }

    if include_optimizer:
        config["optimizer"] = _serialize_optimizer(model.optimizer)

    if metadata:
        config["metadata"] = metadata

    # Save configuration as JSON
    json_path = path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    # Collect all weights
    weights_dict = {}
    for i, layer in enumerate(model.layers):
        params = layer.parameters()
        for j, param in enumerate(params):
            weights_dict[f"layer_{i}_param_{j}"] = to_numpy(param)

        # Save BatchNorm running statistics
        from neura_command.layers import BatchNorm

        if isinstance(layer, BatchNorm):
            weights_dict[f"layer_{i}_running_mean"] = to_numpy(layer.running_mean)
            weights_dict[f"layer_{i}_running_var"] = to_numpy(layer.running_var)

    # Save weights as NPZ
    npz_path = path.with_suffix(".npz")
    np.savez(npz_path, **weights_dict)


def load_model(
    path: Union[str, Path],
    load_optimizer: bool = True,
    device: str = "cpu",
) -> "Sequential":
    """Load a model from disk.

    Args:
        path: Base path of saved model (without extension)
        load_optimizer: Whether to load optimizer configuration
        device: Execution device ('cpu', 'cuda', or 'gpu'). Files are portable.

    Returns:
        Loaded Sequential model

    Example:
        model = load_model('models/my_model')
    """
    from neura_command.network.sequential import Sequential
    from neura_command.backend import DeviceContext, get_backend

    get_backend(device)

    path = Path(path)
    json_path = path.with_suffix(".json")
    npz_path = path.with_suffix(".npz")

    if not json_path.exists():
        raise FileNotFoundError(f"Model config not found: {json_path}")
    if not npz_path.exists():
        raise FileNotFoundError(f"Model weights not found: {npz_path}")

    # Load configuration
    with open(json_path, "r") as f:
        config = json.load(f)

    # Check version compatibility
    format_version = config.get("format_version", "1.0")
    if not format_version.startswith("1."):
        raise ValueError(f"Unsupported model format version: {format_version}")

    # Reconstruct layers
    with DeviceContext("cpu"):
        layers = [
            _deserialize_layer(layer_config)
            for layer_config in config["model"]["layers"]
        ]

    # Reconstruct optimizer and loss
    if load_optimizer and "optimizer" in config:
        optimizer = _deserialize_optimizer(config["optimizer"].copy())
    else:
        from neura_command.network_utils.optimizers import Adam

        optimizer = Adam()

    loss = _deserialize_loss(config["loss"].copy())

    # Create model
    model = Sequential(layers=layers, optimizer=optimizer, loss=loss, device="cpu")

    # Load weights
    with np.load(npz_path, allow_pickle=False) as weights:
        for i, layer in enumerate(model.layers):
            params = layer.parameters()
            for j, param in enumerate(params):
                key = f"layer_{i}_param_{j}"
                if key in weights:
                    param[...] = weights[key]

            # Load BatchNorm running statistics
            from neura_command.layers import BatchNorm

            if isinstance(layer, BatchNorm):
                mean_key = f"layer_{i}_running_mean"
                var_key = f"layer_{i}_running_var"
                if mean_key in weights:
                    layer.running_mean = weights[mean_key]
                if var_key in weights:
                    layer.running_var = weights[var_key]

    return model.to(device)


def get_model_info(path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a saved model without loading weights.

    Args:
        path: Base path of saved model

    Returns:
        Dictionary with model information
    """
    path = Path(path)
    json_path = path.with_suffix(".json")
    npz_path = path.with_suffix(".npz")

    with open(json_path, "r") as f:
        config = json.load(f)

    # Get weights file size
    weights_size = npz_path.stat().st_size if npz_path.exists() else 0

    return {
        "neura_command_version": config.get("neura_command_version"),
        "format_version": config.get("format_version"),
        "saved_at": config.get("saved_at"),
        "num_layers": len(config["model"]["layers"]),
        "layers": [layer["class"] for layer in config["model"]["layers"]],
        "optimizer": config.get("optimizer", {}).get("class"),
        "loss": config.get("loss", {}).get("class"),
        "weights_size_bytes": weights_size,
        "metadata": config.get("metadata"),
    }
