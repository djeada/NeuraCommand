"""
Create network command - create neural network models from JSON architecture files.

Supports both the modern API (Sequential, Dense, ReLU, etc.) and legacy format
for backward compatibility.
"""

import json
import logging
import math
import sys
from typing import Any, Dict, List, Optional

from app.commands.base_command import BaseCommand


class CreateNetworkCommand(BaseCommand):
    """Create a new neural network from a JSON architecture specification."""

    # Supported activation functions
    ACTIVATION_MAP = {
        "relu": "ReLU",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "softmax": "Softmax",
        "leaky_relu": "LeakyReLU",
        "leakyrelu": "LeakyReLU",
        "elu": "ELU",
        "linear": None,
    }

    # Supported optimizers
    OPTIMIZER_CLASSES = ["SGD", "Adam", "RMSprop", "AdaGrad", "Adadelta", "Nadam"]

    # Supported loss functions
    LOSS_CLASSES = ["MSELoss", "CrossEntropyLoss", "HuberLoss", "DifferenceLoss"]

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--seed",
            type=BaseCommand.seed,
            help="Random seed for reproducible initialization.",
        )
        parser.add_argument(
            "--device",
            choices=["cpu", "cuda", "gpu"],
            default="cpu",
            help="Model creation device (default: cpu).",
        )
        parser.add_argument(
            "--architecture",
            "--network-architecture-json",
            type=BaseCommand.file_path,
            required=True,
            dest="architecture",
            help="Path to JSON file describing the neural network architecture.",
        )
        parser.add_argument(
            "--output",
            "--model-pkl",
            type=str,
            default=None,
            dest="output",
            help="Output path for the model. Defaults to 'model' (creates model.json and model.npz).",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["modern", "legacy"],
            default="modern",
            help="Output format: 'modern' (JSON+NPZ, recommended) or 'legacy' (pickle). Default: modern.",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output.",
        )
        parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Suppress all output except errors.",
        )
        parser.add_argument(
            "--validate",
            action="store_true",
            help="Validate the architecture without creating the model.",
        )

    def execute(self):
        # Configure logging based on verbosity
        self._configure_logging()

        # Load architecture JSON
        architecture = self._load_architecture()
        if architecture is None:
            sys.exit(1)

        # Validate the architecture
        if not self._validate_architecture(architecture):
            sys.exit(1)

        # If validate-only mode, exit here
        if getattr(self.args, "validate", False):
            self._log_info("Architecture is valid.")
            return

        # Determine output path
        output_path = self._get_output_path()
        if getattr(self.args, "seed", None) is not None:
            from neura_command import set_random_seed

            set_random_seed(self.args.seed, getattr(self.args, "device", "cpu"))

        # Build and save the model
        if getattr(self.args, "format", "modern") == "legacy":
            self._create_legacy_model(architecture, output_path)
        else:
            self._create_modern_model(architecture, output_path)

    def _configure_logging(self):
        """Configure logging based on verbose/quiet flags."""
        if getattr(self.args, "quiet", False):
            level = logging.ERROR
        elif getattr(self.args, "verbose", False):
            level = logging.DEBUG
        else:
            level = logging.INFO

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def _log_info(self, message: str):
        """Log info message unless quiet mode is enabled."""
        if not getattr(self.args, "quiet", False):
            print(message)

    def _log_error(self, message: str):
        """Log error message."""
        print(f"Error: {message}", file=sys.stderr)

    def _load_architecture(self) -> Optional[Dict[str, Any]]:
        """Load and parse the architecture JSON file."""
        try:
            with open(self.args.architecture, "r") as f:
                self._log_info(f"Loading architecture from {self.args.architecture}")
                data = json.load(f)
                return data
        except json.JSONDecodeError as e:
            self._log_error(f"Invalid JSON in architecture file: {e}")
            return None
        except Exception as e:
            self._log_error(f"Error loading architecture file: {e}")
            return None

    def _validate_architecture(self, arch: Dict[str, Any]) -> bool:
        """Validate the architecture specification."""
        if not isinstance(arch, dict):
            self._log_error("Architecture must be a JSON object")
            return False
        errors = []

        # Check for required fields
        if "layers" not in arch:
            errors.append("Missing required field: 'layers'")
        else:
            layers = arch["layers"]
            if not isinstance(layers, list) or len(layers) == 0:
                errors.append("'layers' must be a non-empty list")
            else:
                for i, layer in enumerate(layers):
                    layer_errors = self._validate_layer(layer, i)
                    errors.extend(layer_errors)

        # Validate optimizer if present
        if "optimizer" in arch:
            opt = arch["optimizer"]
            if isinstance(opt, str):
                opt = {"class": opt}
            if isinstance(opt, dict):
                opt_class = opt.get("class") or opt.get("type")
                if opt_class and opt_class not in self.OPTIMIZER_CLASSES:
                    errors.append(
                        f"Unknown optimizer: '{opt_class}'. "
                        f"Supported: {', '.join(self.OPTIMIZER_CLASSES)}"
                    )
                for key in (
                    "learning_rate",
                    "epsilon",
                    "momentum",
                    "rho",
                    "beta1",
                    "beta2",
                    "weight_decay",
                    "l1_regularization",
                ):
                    if key not in opt:
                        continue
                    value = opt[key]
                    valid = (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                    )
                    if valid:
                        valid = value > 0 if key == "epsilon" else value >= 0
                        if key in {"momentum", "rho", "beta1", "beta2"}:
                            valid = valid and value < 1
                    if not valid:
                        errors.append(f"Optimizer: invalid {key}")
                if "nesterov" in opt and not isinstance(opt["nesterov"], bool):
                    errors.append("Optimizer: nesterov must be boolean")
            else:
                errors.append("Optimizer must be a name or an object")

        # Validate loss if present
        if "loss" in arch:
            loss = arch["loss"]
            if isinstance(loss, str):
                loss = {"class": loss}
            if isinstance(loss, dict):
                loss_class = loss.get("class") or loss.get("type")
                if loss_class and loss_class not in self.LOSS_CLASSES:
                    errors.append(
                        f"Unknown loss function: '{loss_class}'. "
                        f"Supported: {', '.join(self.LOSS_CLASSES)}"
                    )
                if loss_class == "HuberLoss" and "delta" in loss:
                    value = loss["delta"]
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        errors.append("HuberLoss: delta must be finite and positive")
            else:
                errors.append("Loss must be a name or an object")

        if not errors:
            shape = None
            for i, layer in enumerate(arch["layers"]):
                kind = (layer.get("type") or layer.get("class")).lower()
                if kind == "input":
                    value = layer["shape"]
                    shape = (value,) if isinstance(value, int) else tuple(value)
                elif kind == "flatten" and shape is not None:
                    shape = (math.prod(shape),)
                elif kind == "dense":
                    inputs = layer.get("in_features", layer.get("input_size"))
                    if shape is not None and shape != (inputs,):
                        errors.append(
                            f"Layer {i}: Dense expects {inputs} features; previous shape is {shape}. Add Flatten if needed."
                        )
                    shape = (layer.get("out_features", layer.get("output_size")),)
                elif kind == "batchnorm":
                    if shape is not None and shape != (layer["num_features"],):
                        errors.append(
                            f"Layer {i}: BatchNorm feature count does not match previous shape {shape}"
                        )
                    shape = (layer["num_features"],)

        if errors:
            self._log_error("Architecture validation failed:")
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return False

        return True

    def _validate_layer(self, layer: Dict[str, Any], index: int) -> List[str]:
        """Validate a single layer configuration."""
        errors = []
        if not isinstance(layer, dict):
            return [f"Layer {index}: must be a JSON object"]
        layer_type = layer.get("type") or layer.get("class")

        if not isinstance(layer_type, str) or not layer_type:
            errors.append(f"Layer {index}: missing 'type' or 'class' field")
            return errors

        # Validate Dense layer
        if layer_type.lower() == "dense":
            for key, alias in (
                ("in_features", "input_size"),
                ("out_features", "output_size"),
            ):
                if key in layer or alias in layer:
                    value = layer.get(key, layer.get(alias))
                    if type(value) is not int or value < 1:
                        errors.append(
                            f"Layer {index}: {key} must be a positive integer"
                        )
            if "use_bias" in layer and not isinstance(layer["use_bias"], bool):
                errors.append(f"Layer {index}: use_bias must be boolean")
            if layer.get("activation") is not None and (
                not isinstance(layer["activation"], str)
                or layer["activation"].lower() not in self.ACTIVATION_MAP
            ):
                errors.append(f"Layer {index}: unsupported Dense activation")
            if layer.get("weight_init", "he") not in {"he", "xavier"}:
                errors.append(f"Layer {index}: weight_init must be 'he' or 'xavier'")
            if "in_features" not in layer and "input_size" not in layer:
                errors.append(
                    f"Layer {index} (Dense): missing 'in_features' or 'input_size'"
                )
            if "out_features" not in layer and "output_size" not in layer:
                errors.append(
                    f"Layer {index} (Dense): missing 'out_features' or 'output_size'"
                )

        # Validate Dropout layer
        elif layer_type.lower() == "dropout":
            rate = layer.get("rate", 0.5)
            if (
                not isinstance(rate, (int, float))
                or isinstance(rate, bool)
                or not math.isfinite(rate)
                or not (0 <= rate < 1)
            ):
                errors.append(
                    f"Layer {index} (Dropout): rate must be between 0 and 1, got {rate}"
                )

        # Validate BatchNorm layer
        elif layer_type.lower() == "batchnorm":
            if "num_features" not in layer:
                errors.append(f"Layer {index} (BatchNorm): missing 'num_features'")
            elif type(layer["num_features"]) is not int or layer["num_features"] < 1:
                errors.append(f"Layer {index}: num_features must be a positive integer")
            for key, default in (("epsilon", 1e-5), ("momentum", 0.1)):
                value = layer.get(key, default)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or (key == "epsilon" and value <= 0)
                    or (key == "momentum" and not 0 <= value <= 1)
                ):
                    errors.append(f"Layer {index}: invalid BatchNorm {key}")

        elif layer_type.lower() == "flatten":
            pass

        elif layer_type.lower() == "input":
            if "shape" not in layer:
                errors.append(f"Layer {index} (Input): missing 'shape'")
            else:
                shape = layer["shape"]
                shape = [shape] if isinstance(shape, int) else shape
                if (
                    not isinstance(shape, (list, tuple))
                    or not shape
                    or any(type(size) is not int or size < 1 for size in shape)
                ):
                    errors.append(
                        f"Layer {index}: Input shape must contain positive integers"
                    )

        # Validate activation layers
        elif layer_type.lower() in self.ACTIVATION_MAP:
            pass  # Valid activation

        # Validate activation layer with name
        elif layer_type.lower() == "activation":
            activation = layer.get("activation", "")
            activation = activation.lower() if isinstance(activation, str) else ""
            if activation not in self.ACTIVATION_MAP:
                errors.append(
                    f"Layer {index} (Activation): unknown activation '{activation}'. "
                    f"Supported: {', '.join(self.ACTIVATION_MAP.keys())}"
                )

        else:
            errors.append(f"Layer {index}: unknown layer type '{layer_type}'")

        if "alpha" in layer and (
            not isinstance(layer["alpha"], (int, float))
            or isinstance(layer["alpha"], bool)
            or not math.isfinite(layer["alpha"])
        ):
            errors.append(f"Layer {index}: alpha must be finite")

        return errors

    def _get_output_path(self) -> str:
        """Determine the output path for the model."""
        if self.args.output:
            return self.args.output

        # Default based on format
        if getattr(self.args, "format", "modern") == "legacy":
            return "network.pkl"
        else:
            return "model"

    def _create_modern_model(self, arch: Dict[str, Any], output_path: str):
        """Create a model using the modern Sequential API."""
        try:
            from neura_command import Sequential
            from neura_command.serialization.model_io import model_paths

            self._log_info("Building network with modern API...")

            # Build layers
            layers = []
            for layer_config in arch["layers"]:
                layer = self._create_layer(layer_config)
                if layer is not None:
                    layers.append(layer)

            if not layers:
                self._log_error("No valid layers created")
                sys.exit(1)

            # Get optimizer
            optimizer = self._create_optimizer(arch.get("optimizer", {"class": "Adam"}))

            # Get loss function
            loss = self._create_loss(arch.get("loss", {"class": "MSELoss"}))

            # Create the Sequential model
            model = Sequential(
                layers=layers,
                optimizer=optimizer,
                loss=loss,
                device=getattr(self.args, "device", "cpu"),
            )

            # Save the model
            model.save(output_path)
            config_path, weights_path = model_paths(output_path)

            self._log_info("Model created successfully!")
            self._log_info(f"  Architecture: {config_path}")
            self._log_info(f"  Weights: {weights_path}")
            self._log_info(f"  Total parameters: {model.num_parameters:,}")

            # Show model summary if verbose
            if getattr(self.args, "verbose", False):
                print("\nModel Summary:")
                print(model.summary())

        except Exception as e:
            self._log_error(f"Error creating model: {e}")
            if getattr(self.args, "verbose", False):
                import traceback

                traceback.print_exc()
            sys.exit(1)

    def _create_layer(self, config: Dict[str, Any]):
        """Create a layer from configuration."""
        from neura_command import (
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

        layer_type = (config.get("type") or config.get("class", "")).lower()

        if layer_type == "dense":
            in_features = config.get("in_features") or config.get("input_size")
            out_features = config.get("out_features") or config.get("output_size")
            use_bias = config.get("use_bias", True)
            activation = self._create_dense_activation(
                config.get("activation"),
                alpha=config.get("alpha"),
            )
            return Dense(
                in_features,
                out_features,
                activation=activation,
                use_bias=use_bias,
                weight_init=config.get("weight_init", "he"),
            )

        elif layer_type == "dropout":
            rate = config.get("rate", 0.5)
            return Dropout(rate)

        elif layer_type == "batchnorm":
            num_features = config["num_features"]
            return BatchNorm(
                num_features,
                epsilon=config.get("epsilon", 1e-5),
                momentum=config.get("momentum", 0.1),
            )

        elif layer_type == "flatten":
            return Flatten()

        elif layer_type == "input":
            shape = config["shape"]
            return Input(tuple(shape) if isinstance(shape, list) else shape)

        elif layer_type == "relu":
            return ReLU()

        elif layer_type == "sigmoid":
            return Sigmoid()

        elif layer_type == "tanh":
            return Tanh()

        elif layer_type == "softmax":
            return Softmax()

        elif layer_type in ("leakyrelu", "leaky_relu"):
            alpha = config.get("alpha", 0.01)
            return LeakyReLU(alpha)

        elif layer_type == "elu":
            alpha = config.get("alpha", 1.0)
            return ELU(alpha)

        elif layer_type == "activation":
            return Activation(
                self._create_dense_activation(
                    config.get("activation", "relu"), alpha=config.get("alpha")
                )
            )

        elif layer_type == "linear":
            return Activation("linear")

        else:
            self._log_error(f"Unknown layer type: {layer_type}")
            return None

    def _create_dense_activation(
        self,
        activation_name: Optional[str],
        alpha: Optional[float] = None,
    ):
        """Create an activation function instance for Dense layers."""
        if not activation_name:
            return None

        from neura_command.network_utils.activation_functions import (
            ELUActivation,
            LeakyReLUActivation,
            LinearActivation,
            ReLUActivation,
            SigmoidActivation,
            SoftmaxActivation,
            TanhActivation,
        )

        activation_key = activation_name.lower()
        activation_map = {
            "relu": ReLUActivation,
            "sigmoid": SigmoidActivation,
            "tanh": TanhActivation,
            "softmax": SoftmaxActivation,
            "linear": LinearActivation,
            "leaky_relu": LeakyReLUActivation,
            "leakyrelu": LeakyReLUActivation,
            "elu": ELUActivation,
        }

        if activation_key not in activation_map:
            self._log_error(f"Unknown dense activation: {activation_name}")
            return None

        activation_class = activation_map[activation_key]
        if activation_key in ("leaky_relu", "leakyrelu", "elu"):
            if alpha is None:
                alpha = 0.01 if "leaky" in activation_key else 1.0
            return activation_class(alpha)
        return activation_class()

    def _create_optimizer(self, config: Dict[str, Any]):
        """Create optimizer from configuration."""
        from neura_command import SGD, Adam, RMSprop, AdaGrad, Adadelta, Nadam

        if isinstance(config, str):
            config = {"class": config}

        opt_class = config.get("class") or config.get("type", "Adam")
        lr = config.get("learning_rate", 0.001)

        if opt_class == "SGD":
            return SGD(
                learning_rate=lr,
                momentum=config.get("momentum", 0.0),
                nesterov=config.get("nesterov", False),
                weight_decay=config.get("weight_decay", 0.0),
                l1_regularization=config.get("l1_regularization", 0.0),
            )
        elif opt_class in ("Adam", "Nadam"):
            optimizer_class = Adam if opt_class == "Adam" else Nadam
            return optimizer_class(
                learning_rate=lr,
                beta1=config.get("beta1", 0.9),
                beta2=config.get("beta2", 0.999),
                epsilon=config.get("epsilon", 1e-8),
                weight_decay=config.get("weight_decay", 0.0),
            )
        elif opt_class == "RMSprop":
            return RMSprop(
                learning_rate=lr,
                rho=config.get("rho", 0.9),
                epsilon=config.get("epsilon", 1e-8),
                weight_decay=config.get("weight_decay", 0.0),
            )
        elif opt_class == "AdaGrad":
            return AdaGrad(learning_rate=lr, epsilon=config.get("epsilon", 1e-8))
        elif opt_class == "Adadelta":
            return Adadelta(
                learning_rate=config.get("learning_rate", 1.0),
                rho=config.get("rho", 0.95),
                epsilon=config.get("epsilon", 1e-6),
                weight_decay=config.get("weight_decay", 0.0),
            )
        else:
            self._log_info(f"Unknown optimizer '{opt_class}', using Adam")
            return Adam(learning_rate=lr)

    def _create_loss(self, config: Dict[str, Any]):
        """Create loss function from configuration."""
        from neura_command import MSELoss, CrossEntropyLoss, HuberLoss
        from neura_command.network_utils.loss_functions import DifferenceLoss

        if isinstance(config, str):
            config = {"class": config}

        loss_class = config.get("class") or config.get("type", "MSELoss")

        if loss_class == "MSELoss":
            return MSELoss()
        elif loss_class == "CrossEntropyLoss":
            return CrossEntropyLoss()
        elif loss_class == "HuberLoss":
            return HuberLoss(delta=config.get("delta", 1.0))
        elif loss_class == "DifferenceLoss":
            return DifferenceLoss()
        else:
            self._log_info(f"Unknown loss '{loss_class}', using MSELoss")
            return MSELoss()

    def _create_legacy_model(self, arch: Dict[str, Any], output_path: str):
        """Create a model using the legacy API (pickle format)."""
        try:
            if getattr(self.args, "device", "cpu") != "cpu":
                raise ValueError(
                    "Legacy models support CPU only; use --format modern for CUDA"
                )
            # Import legacy modules
            from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
                FeedForwardNetworkIO,
            )
            from neura_command.deprecated.network_builder.network_builder import (
                NetworkBuilder,
            )

            self._log_info("Building network with legacy API...")

            network_builder = NetworkBuilder.from_json(json_config=arch)
            model = network_builder.build()
            if not output_path.endswith(".pkl"):
                output_path += ".pkl"

            FeedForwardNetworkIO.save_to_pickle(model, output_path)
            self._log_info(f"Model saved to {output_path}")

        except Exception as e:
            self._log_error(f"Error creating legacy model: {e}")
            if getattr(self.args, "verbose", False):
                import traceback

                traceback.print_exc()
            sys.exit(1)
