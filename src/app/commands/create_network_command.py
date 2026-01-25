"""
Create network command - create neural network models from JSON architecture files.

Supports both the modern API (Sequential, Dense, ReLU, etc.) and legacy format
for backward compatibility.
"""

import json
import logging
import sys
from pathlib import Path
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
        "elu": "ELU",
        "linear": None,
    }

    # Supported optimizers
    OPTIMIZER_CLASSES = ["SGD", "Adam", "RMSprop", "AdaGrad"]

    # Supported loss functions
    LOSS_CLASSES = ["MSELoss", "CrossEntropyLoss", "HuberLoss"]

    @staticmethod
    def add_arguments(parser):
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
        if getattr(self.args, 'validate', False):
            self._log_info("Architecture is valid.")
            return

        # Determine output path
        output_path = self._get_output_path()

        # Build and save the model
        if getattr(self.args, 'format', 'modern') == "legacy":
            self._create_legacy_model(architecture, output_path)
        else:
            self._create_modern_model(architecture, output_path)

    def _configure_logging(self):
        """Configure logging based on verbose/quiet flags."""
        if getattr(self.args, 'quiet', False):
            level = logging.ERROR
        elif getattr(self.args, 'verbose', False):
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
        if not getattr(self.args, 'quiet', False):
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
            if isinstance(opt, dict):
                opt_class = opt.get("class") or opt.get("type")
                if opt_class and opt_class not in self.OPTIMIZER_CLASSES:
                    errors.append(
                        f"Unknown optimizer: '{opt_class}'. "
                        f"Supported: {', '.join(self.OPTIMIZER_CLASSES)}"
                    )

        # Validate loss if present
        if "loss" in arch:
            loss = arch["loss"]
            if isinstance(loss, dict):
                loss_class = loss.get("class") or loss.get("type")
                if loss_class and loss_class not in self.LOSS_CLASSES:
                    errors.append(
                        f"Unknown loss function: '{loss_class}'. "
                        f"Supported: {', '.join(self.LOSS_CLASSES)}"
                    )

        if errors:
            self._log_error("Architecture validation failed:")
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            return False

        return True

    def _validate_layer(self, layer: Dict[str, Any], index: int) -> List[str]:
        """Validate a single layer configuration."""
        errors = []
        layer_type = layer.get("type") or layer.get("class")

        if not layer_type:
            errors.append(f"Layer {index}: missing 'type' or 'class' field")
            return errors

        # Validate Dense layer
        if layer_type.lower() == "dense":
            if "in_features" not in layer and "input_size" not in layer:
                errors.append(f"Layer {index} (Dense): missing 'in_features' or 'input_size'")
            if "out_features" not in layer and "output_size" not in layer:
                errors.append(f"Layer {index} (Dense): missing 'out_features' or 'output_size'")

        # Validate Dropout layer
        elif layer_type.lower() == "dropout":
            rate = layer.get("rate", 0.5)
            if not (0 <= rate < 1):
                errors.append(f"Layer {index} (Dropout): rate must be between 0 and 1, got {rate}")

        # Validate BatchNorm layer
        elif layer_type.lower() == "batchnorm":
            if "num_features" not in layer:
                errors.append(f"Layer {index} (BatchNorm): missing 'num_features'")

        # Validate activation layers
        elif layer_type.lower() in self.ACTIVATION_MAP:
            pass  # Valid activation

        # Validate activation layer with name
        elif layer_type.lower() == "activation":
            activation = layer.get("activation", "").lower()
            if activation not in self.ACTIVATION_MAP:
                errors.append(
                    f"Layer {index} (Activation): unknown activation '{activation}'. "
                    f"Supported: {', '.join(self.ACTIVATION_MAP.keys())}"
                )

        else:
            errors.append(f"Layer {index}: unknown layer type '{layer_type}'")

        return errors

    def _get_output_path(self) -> str:
        """Determine the output path for the model."""
        if self.args.output:
            return self.args.output

        # Default based on format
        if getattr(self.args, 'format', 'modern') == "legacy":
            return "network.pkl"
        else:
            return "model"

    def _create_modern_model(self, arch: Dict[str, Any], output_path: str):
        """Create a model using the modern Sequential API."""
        try:
            from neura_command import (
                Sequential, Dense, Dropout, BatchNorm, Flatten,
                ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU,
                SGD, Adam, RMSprop, AdaGrad,
                MSELoss, CrossEntropyLoss, HuberLoss,
            )

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
            model = Sequential(layers=layers, optimizer=optimizer, loss=loss)

            # Save the model
            model.save(output_path)

            self._log_info(f"Model created successfully!")
            self._log_info(f"  Architecture: {output_path}.json")
            self._log_info(f"  Weights: {output_path}.npz")
            self._log_info(f"  Total parameters: {model.num_parameters:,}")

            # Show model summary if verbose
            if getattr(self.args, 'verbose', False):
                print("\nModel Summary:")
                print(model.summary())

        except Exception as e:
            self._log_error(f"Error creating model: {e}")
            if getattr(self.args, 'verbose', False):
                import traceback
                traceback.print_exc()
            sys.exit(1)

    def _create_layer(self, config: Dict[str, Any]):
        """Create a layer from configuration."""
        from neura_command import (
            Dense, Dropout, BatchNorm, Flatten,
            ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU,
        )

        layer_type = (config.get("type") or config.get("class", "")).lower()

        if layer_type == "dense":
            in_features = config.get("in_features") or config.get("input_size")
            out_features = config.get("out_features") or config.get("output_size")
            use_bias = config.get("use_bias", True)
            return Dense(in_features, out_features, use_bias=use_bias)

        elif layer_type == "dropout":
            rate = config.get("rate", 0.5)
            return Dropout(rate)

        elif layer_type == "batchnorm":
            num_features = config["num_features"]
            return BatchNorm(num_features)

        elif layer_type == "flatten":
            return Flatten()

        elif layer_type == "relu":
            return ReLU()

        elif layer_type == "sigmoid":
            return Sigmoid()

        elif layer_type == "tanh":
            return Tanh()

        elif layer_type == "softmax":
            return Softmax()

        elif layer_type == "leakyrelu":
            alpha = config.get("alpha", 0.01)
            return LeakyReLU(alpha)

        elif layer_type == "elu":
            alpha = config.get("alpha", 1.0)
            return ELU(alpha)

        elif layer_type == "activation":
            activation = config.get("activation", "relu").lower()
            # Extract alpha value before using in activation map
            alpha = config.get("alpha", 0.01 if activation == "leaky_relu" else 1.0)
            activation_map = {
                "relu": ReLU,
                "sigmoid": Sigmoid,
                "tanh": Tanh,
                "softmax": Softmax,
                "leaky_relu": LeakyReLU,
                "elu": ELU,
            }
            activation_class = activation_map.get(activation, ReLU)
            if activation in ("leaky_relu", "elu"):
                return activation_class(alpha)
            return activation_class()

        else:
            self._log_error(f"Unknown layer type: {layer_type}")
            return None

    def _create_optimizer(self, config: Dict[str, Any]):
        """Create optimizer from configuration."""
        from neura_command import SGD, Adam, RMSprop, AdaGrad

        if isinstance(config, str):
            config = {"class": config}

        opt_class = config.get("class") or config.get("type", "Adam")
        lr = config.get("learning_rate", 0.001)

        if opt_class == "SGD":
            return SGD(
                learning_rate=lr,
                momentum=config.get("momentum", 0.0),
                nesterov=config.get("nesterov", False),
            )
        elif opt_class == "Adam":
            return Adam(
                learning_rate=lr,
                beta1=config.get("beta1", 0.9),
                beta2=config.get("beta2", 0.999),
            )
        elif opt_class == "RMSprop":
            return RMSprop(
                learning_rate=lr,
                rho=config.get("rho", 0.9),
            )
        elif opt_class == "AdaGrad":
            return AdaGrad(learning_rate=lr)
        else:
            self._log_info(f"Unknown optimizer '{opt_class}', using Adam")
            return Adam(learning_rate=lr)

    def _create_loss(self, config: Dict[str, Any]):
        """Create loss function from configuration."""
        from neura_command import MSELoss, CrossEntropyLoss, HuberLoss

        if isinstance(config, str):
            config = {"class": config}

        loss_class = config.get("class") or config.get("type", "MSELoss")

        if loss_class == "MSELoss":
            return MSELoss()
        elif loss_class == "CrossEntropyLoss":
            return CrossEntropyLoss()
        elif loss_class == "HuberLoss":
            return HuberLoss(delta=config.get("delta", 1.0))
        else:
            self._log_info(f"Unknown loss '{loss_class}', using MSELoss")
            return MSELoss()

    def _create_legacy_model(self, arch: Dict[str, Any], output_path: str):
        """Create a model using the legacy API (pickle format)."""
        try:
            # Import legacy modules
            from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
                FeedForwardNetworkIO,
            )
            from neura_command.deprecated.network_builder.network_builder import NetworkBuilder

            self._log_info("Building network with legacy API...")

            network_builder = NetworkBuilder.from_json(json_config=arch)
            model = network_builder.build()

            FeedForwardNetworkIO.save_to_pickle(model, output_path)
            self._log_info(f"Model saved to {output_path}")

        except Exception as e:
            self._log_error(f"Error creating legacy model: {e}")
            if getattr(self.args, 'verbose', False):
                import traceback
                traceback.print_exc()
            sys.exit(1)
