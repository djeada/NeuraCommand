"""
Train network command - train neural network models.

Supports both modern Sequential models and legacy pickle models,
with options for learning rate, batch size, validation split, and more.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

from app.commands.base_command import BaseCommand


class TrainNetworkCommand(BaseCommand):
    """Train an existing neural network model."""

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--seed",
            type=BaseCommand.seed,
            help="Seed validation splitting, shuffling, and dropout.",
        )
        parser.add_argument(
            "--no-shuffle",
            action="store_true",
            help="Keep input order for validation splitting and training epochs.",
        )
        parser.add_argument(
            "--device",
            choices=["cpu", "cuda", "gpu"],
            default="cpu",
            help="Training device (CUDA requires optional CuPy and a working GPU).",
        )
        # Model paths
        parser.add_argument(
            "--model",
            "--model-pkl",
            type=str,
            required=True,
            dest="model",
            help="Path to the model (without extension for modern format, or .pkl for legacy).",
        )
        parser.add_argument(
            "--features-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to CSV file containing input features for training.",
        )
        parser.add_argument(
            "--targets-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to CSV file containing target values.",
        )

        # Training parameters
        parser.add_argument(
            "--epochs",
            type=BaseCommand.positive_int,
            default=100,
            help="Number of training epochs (default: 100).",
        )
        parser.add_argument(
            "--batch-size",
            type=BaseCommand.positive_int,
            default=32,
            help="Batch size for training (default: 32).",
        )
        parser.add_argument(
            "--learning-rate",
            "--lr",
            type=BaseCommand.positive_float,
            default=None,
            dest="learning_rate",
            help="Learning rate (overrides model's optimizer learning rate).",
        )
        parser.add_argument(
            "--validation-split",
            type=BaseCommand.validation_fraction,
            default=0.0,
            help="Fraction of data to use for validation in [0, 1), default: 0.0.",
        )

        # Output options
        parser.add_argument(
            "--output",
            "--output-model-pkl",
            type=str,
            default=None,
            dest="output",
            help="Path to save the trained model. Required unless --overwrite is given.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite the input model file (ignored if --output is specified).",
        )
        parser.add_argument(
            "--save-history",
            type=str,
            help="Path to save training history as JSON for visualization.",
        )

        # Display options
        parser.add_argument(
            "--show-summary",
            action="store_true",
            help="Display training summary after training completes.",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output during training.",
        )
        parser.add_argument(
            "--quiet",
            "-q",
            action="store_true",
            help="Suppress all output except errors.",
        )

        # Early stopping
        parser.add_argument(
            "--early-stopping",
            type=BaseCommand.positive_int,
            default=None,
            metavar="PATIENCE",
            help="Enable early stopping with specified patience (epochs).",
        )

    def execute(self):
        # Configure logging
        self._configure_logging()
        if getattr(self.args, "seed", None) is not None:
            from neura_command import set_random_seed

            set_random_seed(self.args.seed, getattr(self.args, "device", "cpu"))

        if not self._has_output_destination():
            self._log_error("Output path not specified. Use --output or --overwrite.")
            sys.exit(1)

        # Load the model
        model = self._load_model()
        if model is None:
            sys.exit(1)

        # Update learning rate if specified
        if self.args.learning_rate is not None:
            self._set_learning_rate(model, self.args.learning_rate)

        # Load training data
        features, targets = self._load_training_data()
        if features is None or targets is None:
            sys.exit(1)

        # Validate data shapes
        if not self._validate_data(features, targets):
            sys.exit(1)

        # Prepare validation data if requested
        (
            train_features,
            train_targets,
            val_features,
            val_targets,
        ) = self._split_validation_data(features, targets)

        # Set up callbacks
        callbacks = self._setup_callbacks()

        # Train the model
        history = self._train_model(
            model, train_features, train_targets, val_features, val_targets, callbacks
        )
        if history is None:
            sys.exit(1)

        # Save training history if requested
        if self.args.save_history:
            self._save_history(history)

        # Show training summary if requested
        if self.args.show_summary:
            self._display_training_summary(history)

        # Save the trained model
        self._save_model(model)

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

    def _load_model(self):
        """Load the model from disk."""
        model_path = self.args.model

        # Try modern format first
        try:
            from neura_command.serialization import load_model

            model = load_model(model_path, device=getattr(self.args, "device", "cpu"))
            self._log_info(f"Loaded model from {model_path} (modern format)")
            self._model_format = "modern"
            return model
        except FileNotFoundError:
            pass
        except Exception as e:
            self._log_error(f"Error loading modern model: {e}")
            return None

        # Try legacy pickle format
        if model_path.endswith(".pkl") or Path(model_path + ".pkl").exists():
            try:
                if getattr(self.args, "device", "cpu") != "cpu":
                    raise ValueError(
                        "Legacy models support CPU only; create a modern model for CUDA"
                    )
                from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
                    FeedForwardNetworkIO,
                )

                pkl_path = (
                    model_path if model_path.endswith(".pkl") else model_path + ".pkl"
                )
                model = FeedForwardNetworkIO.load_from_pickle(pkl_path)
                self._log_info(f"Loaded model from {pkl_path} (legacy format)")
                self._model_format = "legacy"
                return model
            except Exception as e:
                self._log_error(f"Error loading legacy model: {e}")
                return None

        self._log_error(f"Model not found at '{model_path}'")
        self._log_error("  Expected files: {path}.json and {path}.npz (modern)")
        self._log_error("  Or: {path}.pkl (legacy)")
        return None

    def _set_learning_rate(self, model, lr: float):
        """Update the model's optimizer learning rate."""
        if hasattr(model, "optimizer") and hasattr(model.optimizer, "learning_rate"):
            old_lr = model.optimizer.learning_rate
            model.optimizer.learning_rate = lr
            self._log_info(f"Learning rate changed: {old_lr} -> {lr}")
        else:
            self.logger.warning(
                "Could not update learning rate (model may not support it)"
            )

    def _load_training_data(self):
        """Load training features and targets from CSV files."""
        try:
            features = np.loadtxt(
                self.args.features_csv, delimiter=",", dtype=np.float32, ndmin=2
            )
            self._log_info(f"Loaded features: shape {features.shape}")
        except Exception as e:
            self._log_error(f"Error loading features: {e}")
            return None, None

        try:
            targets = np.loadtxt(
                self.args.targets_csv, delimiter=",", dtype=np.float32, ndmin=2
            )
            self._log_info(f"Loaded targets: shape {targets.shape}")
        except Exception as e:
            self._log_error(f"Error loading targets: {e}")
            return None, None

        return features, targets

    def _validate_data(self, features: np.ndarray, targets: np.ndarray) -> bool:
        """Validate that data shapes are compatible."""
        # Check sample counts match (works for both 1D and 2D)
        if len(features) != len(targets):
            self._log_error(
                f"Sample count mismatch: features has {len(features)} samples, "
                f"targets has {len(targets)} samples"
            )
            return False

        if (
            len(features) == 0
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(targets))
        ):
            self._log_error("Training data must be non-empty and finite")
            return False

        return True

    def _split_validation_data(self, features: np.ndarray, targets: np.ndarray):
        """Split off validation data if requested."""
        val_split = self.args.validation_split

        if val_split <= 0:
            return features, targets, None, None

        n_samples = len(features)
        if n_samples < 2:
            self._log_error("Validation splitting requires at least two samples")
            raise SystemExit(1)
        n_val = min(n_samples - 1, max(1, int(n_samples * val_split)))

        if getattr(self.args, "no_shuffle", False):
            train_indices = np.arange(n_samples - n_val)
            val_indices = np.arange(n_samples - n_val, n_samples)
        else:
            indices = np.random.permutation(n_samples)
            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

        val_features = features[val_indices]
        val_targets = targets[val_indices]
        train_features = features[train_indices]
        train_targets = targets[train_indices]

        self._log_info(
            f"Using {n_val} samples for validation, {len(train_indices)} for training"
        )

        return train_features, train_targets, val_features, val_targets

    def _has_output_destination(self) -> bool:
        """Whether training has a valid destination for the trained model."""
        return bool(self.args.output or self.args.overwrite)

    def _setup_callbacks(self):
        """Set up training callbacks."""
        callbacks = []

        # Early stopping
        if self.args.early_stopping is not None:
            try:
                from neura_command.network_utils.callbacks import EarlyStopping

                callbacks.append(
                    EarlyStopping(
                        patience=self.args.early_stopping,
                        monitor="val_loss"
                        if self.args.validation_split > 0
                        else "loss",
                        min_delta=1e-4,
                    )
                )
                self._log_info(
                    f"Early stopping enabled with patience={self.args.early_stopping}"
                )
            except ImportError:
                self.logger.warning("Early stopping not available")

        return callbacks if callbacks else None

    def _train_model(
        self, model, features, targets, val_features, val_targets, callbacks
    ):
        """Train the model."""
        self._log_info(
            f"Training for {self.args.epochs} epochs, batch size {self.args.batch_size}"
        )

        try:
            # Ensure data is 2D
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)

            # Train the model
            train_kwargs = {
                "epochs": self.args.epochs,
                "batch_size": self.args.batch_size,
                "verbose": getattr(self.args, "verbose", False),
            }
            import inspect

            train_parameters = inspect.signature(model.train).parameters
            if "seed" in train_parameters:
                train_kwargs["seed"] = getattr(self.args, "seed", None)
            if "shuffle" in train_parameters:
                train_kwargs["shuffle"] = not getattr(self.args, "no_shuffle", False)

            # Add validation data if the model supports it
            if val_features is not None:
                import inspect

                train_sig = inspect.signature(model.train)
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in train_sig.parameters.values()
                )
                if val_features.ndim == 1:
                    val_features = val_features.reshape(-1, 1)
                if val_targets.ndim == 1:
                    val_targets = val_targets.reshape(-1, 1)
                if "validation_data" in train_sig.parameters or accepts_kwargs:
                    train_kwargs["validation_data"] = (val_features, val_targets)
                elif "X_val" in train_sig.parameters:
                    train_kwargs["X_val"] = val_features
                    train_kwargs["y_val"] = val_targets

            # Add callbacks if the model supports them
            if callbacks and hasattr(model, "train"):
                import inspect

                train_sig = inspect.signature(model.train)
                if "callbacks" in train_sig.parameters:
                    train_kwargs["callbacks"] = callbacks

            history = model.train(features, targets, **train_kwargs)

            if history is None:
                history = {"epochs": self.args.epochs}

            self._log_info("Training completed!")
            return history

        except Exception as e:
            self._log_error(f"Error during training: {e}")
            if getattr(self.args, "verbose", False):
                import traceback

                traceback.print_exc()
            return None

    def _save_history(self, history):
        """Save training history to JSON file."""
        if history is None:
            self.logger.warning("No history to save")
            return

        try:
            # Convert numpy types to Python types
            serializable_history = {}
            for key, value in history.items():
                if isinstance(value, (list, tuple)):
                    serializable_history[key] = [
                        float(v) if hasattr(v, "item") else v for v in value
                    ]
                elif hasattr(value, "item"):
                    serializable_history[key] = float(value)
                else:
                    serializable_history[key] = value

            Path(self.args.save_history).parent.mkdir(parents=True, exist_ok=True)
            with open(self.args.save_history, "w") as f:
                json.dump(serializable_history, f, indent=2)
            self._log_info(f"Training history saved to {self.args.save_history}")
        except Exception as e:
            self.logger.warning(f"Could not save training history: {e}")

    def _display_training_summary(self, history):
        """Display a summary of training progress."""
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)

        if history is None:
            print(f"Epochs completed: {self.args.epochs}")
            print("(Detailed history not available)")
            print("=" * 50)
            return

        if isinstance(history, dict):
            for metric, values in history.items():
                if isinstance(values, list) and len(values) > 0:
                    print(f"\n{metric}:")
                    print(f"  Initial: {values[0]:.6f}")
                    print(f"  Final:   {values[-1]:.6f}")

                    if "loss" in metric.lower():
                        print(f"  Best:    {min(values):.6f}")
                    else:
                        print(f"  Best:    {max(values):.6f}")

                    if len(values) >= 2 and values[0] != 0:
                        change = values[0] - values[-1]
                        pct = abs(change) / abs(values[0]) * 100
                        direction = "decreased" if change > 0 else "increased"
                        print(f"  Change:  {direction} by {pct:.1f}%")

        print("\n" + "=" * 50)

    def _save_model(self, model):
        """Save the trained model."""
        # Determine output path
        if self.args.output:
            output_path = self.args.output
        elif self.args.overwrite:
            output_path = self.args.model
        else:
            self._log_error("Output path not specified. Use --output or --overwrite.")
            return

        try:
            if self._model_format == "legacy":
                from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
                    FeedForwardNetworkIO,
                )

                if not output_path.endswith(".pkl"):
                    output_path += ".pkl"
                FeedForwardNetworkIO.save_to_pickle(model, output_path)
            else:
                model.save(output_path)

            self._log_info(f"Model saved to {output_path}")
        except Exception as e:
            self._log_error(f"Error saving model: {e}")
            raise SystemExit(1) from e
