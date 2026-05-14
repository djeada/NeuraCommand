"""
Predict command - make predictions using trained neural network models.

Supports batch predictions, confidence scores, and multiple output formats.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from app.commands.base_command import BaseCommand


class PredictCommand(BaseCommand):
    """Make predictions using a trained neural network model."""

    @staticmethod
    def add_arguments(parser):
        # Model and input
        parser.add_argument(
            "--model",
            "--model-pkl",
            type=str,
            required=True,
            dest="model",
            help="Path to the model (without extension for modern format, or .pkl for legacy).",
        )
        parser.add_argument(
            "--input-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to input CSV file with features for prediction.",
        )

        # Output options
        parser.add_argument(
            "--output-mode",
            type=str,
            choices=["display", "save", "json"],
            default="display",
            help="Output mode: 'display' (stdout), 'save' (CSV), or 'json'. Default: display.",
        )
        parser.add_argument(
            "--output-csv",
            type=str,
            help="Path to save predictions as CSV (required if output-mode is 'save').",
        )
        parser.add_argument(
            "--output-json",
            type=str,
            help="Path to save predictions as JSON (alternative to --output-csv).",
        )

        # Prediction options
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help="Batch size for prediction (useful for large datasets).",
        )
        parser.add_argument(
            "--show-confidence",
            action="store_true",
            help="Show confidence scores for classification predictions.",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=1,
            help="Show top K predictions for classification (default: 1).",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Classification threshold for binary predictions (default: 0.5).",
        )

        # Display options
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
            help="Suppress all output except predictions/errors.",
        )
        parser.add_argument(
            "--precision",
            type=int,
            default=6,
            help="Decimal precision for output values (default: 6).",
        )

    def execute(self):
        # Configure logging
        self._configure_logging()

        # Load the model
        model = self._load_model()
        if model is None:
            sys.exit(1)

        # Load input data
        input_data = self._load_input_data()
        if input_data is None:
            sys.exit(1)

        # Make predictions
        predictions = self._make_predictions(model, input_data)
        if predictions is None:
            sys.exit(1)

        # Output predictions
        self._output_predictions(predictions, input_data)

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
            print(message, file=sys.stderr)

    def _log_error(self, message: str):
        """Log error message."""
        print(f"Error: {message}", file=sys.stderr)

    def _load_model(self):
        """Load the model from disk."""
        model_path = self.args.model

        # Try modern format first
        try:
            from neura_command.serialization import load_model
            model = load_model(model_path)
            self._log_info(f"Loaded model from {model_path}")
            return model
        except FileNotFoundError:
            pass
        except Exception as e:
            self._log_error(f"Error loading modern model: {e}")
            return None

        # Try legacy pickle format
        if model_path.endswith('.pkl') or Path(model_path + '.pkl').exists():
            try:
                from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
                    FeedForwardNetworkIO,
                )
                pkl_path = model_path if model_path.endswith('.pkl') else model_path + '.pkl'
                model = FeedForwardNetworkIO.load_from_pickle(pkl_path)
                self._log_info(f"Loaded model from {pkl_path} (legacy format)")
                return model
            except Exception as e:
                self._log_error(f"Error loading legacy model: {e}")
                return None

        self._log_error(f"Model not found at '{model_path}'")
        return None

    def _load_input_data(self) -> Optional[np.ndarray]:
        """Load input data from CSV file."""
        try:
            data = np.loadtxt(self.args.input_csv, delimiter=",")
            
            # Ensure 2D: 1D array could be single sample with multiple features
            # or multiple samples with single feature. Default to multiple samples.
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self._log_info(f"Loaded {len(data)} samples from {self.args.input_csv}")
            return data
        except Exception as e:
            self._log_error(f"Error loading input data: {e}")
            return None

    def _make_predictions(self, model, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions using the model."""
        try:
            batch_size = self.args.batch_size

            if batch_size is not None and len(input_data) > batch_size:
                # Batch prediction for large datasets
                self._log_info(f"Predicting in batches of {batch_size}...")
                predictions = []
                for i in range(0, len(input_data), batch_size):
                    batch = input_data[i:i + batch_size]
                    batch_pred = np.asarray(model.predict(batch))
                    if batch_pred.ndim == 0:
                        batch_pred = batch_pred.reshape(1, 1)
                    elif batch_pred.ndim == 1:
                        batch_pred = batch_pred.reshape(-1, 1)
                    predictions.append(batch_pred)
                    
                    if getattr(self.args, 'verbose', False):
                        progress = min(i + batch_size, len(input_data))
                        print(f"  Processed {progress}/{len(input_data)} samples", 
                              file=sys.stderr)

                predictions = np.concatenate(predictions, axis=0)
            else:
                predictions = model.predict(input_data)

            self._log_info(f"Generated predictions: shape {predictions.shape}")
            return predictions

        except Exception as e:
            self._log_error(f"Error making predictions: {e}")
            if getattr(self.args, 'verbose', False):
                import traceback
                traceback.print_exc()
            return None

    def _output_predictions(self, predictions: np.ndarray, input_data: np.ndarray):
        """Output predictions in the requested format."""
        output_mode = self.args.output_mode
        precision = self.args.precision

        # Ensure 2D
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        # Add confidence/classification info if requested
        output_data = self._prepare_output_data(predictions)

        if output_mode == "display":
            self._display_predictions(output_data, precision)
        elif output_mode == "save":
            self._save_csv(output_data, precision)
        elif output_mode == "json":
            self._save_json(output_data, input_data, precision)

    def _prepare_output_data(self, predictions: np.ndarray) -> dict:
        """Prepare output data with optional confidence scores."""
        result = {
            "predictions": predictions,
            "raw_values": predictions.copy(),
        }

        # For classification with confidence
        if self.args.show_confidence:
            if predictions.shape[1] > 1:
                # Multi-class: softmax-like output
                result["predicted_classes"] = np.argmax(predictions, axis=1)
                result["confidence"] = np.max(predictions, axis=1)
                
                # Top-K predictions
                if self.args.top_k > 1:
                    k = min(self.args.top_k, predictions.shape[1])
                    result["top_k_classes"] = np.argsort(predictions, axis=1)[:, -k:][:, ::-1]
                    result["top_k_confidence"] = np.sort(predictions, axis=1)[:, -k:][:, ::-1]
            else:
                # Binary classification
                threshold = self.args.threshold
                result["predicted_classes"] = (predictions >= threshold).astype(int).flatten()
                result["confidence"] = np.where(
                    predictions >= threshold,
                    predictions,
                    1 - predictions
                ).flatten()

        return result

    def _display_predictions(self, output_data: dict, precision: int):
        """Display predictions to stdout."""
        predictions = output_data["predictions"]
        
        if self.args.show_confidence and "predicted_classes" in output_data:
            print("\n" + "=" * 60)
            print("Predictions with Confidence")
            print("=" * 60)
            
            classes = output_data["predicted_classes"]
            confidence = output_data["confidence"]
            
            for i, (cls, conf) in enumerate(zip(classes, confidence)):
                raw = predictions[i]
                if predictions.shape[1] <= 4:
                    raw_str = ", ".join(f"{v:.{precision}f}" for v in raw)
                    print(f"Sample {i + 1}: Class {cls} (confidence: {conf:.{precision}f}) | Raw: [{raw_str}]")
                else:
                    print(f"Sample {i + 1}: Class {cls} (confidence: {conf:.{precision}f})")
            
            print("=" * 60)
        else:
            # Simple output
            print("Predictions:")
            np.set_printoptions(precision=precision, suppress=True)
            print(predictions)

    def _save_csv(self, output_data: dict, precision: int):
        """Save predictions to CSV file."""
        output_path = self.args.output_csv
        if not output_path:
            self._log_error("Output CSV path is required for 'save' mode. Use --output-csv.")
            return

        try:
            predictions = output_data["predictions"]
            
            fmt = f'%.{precision}f'
            np.savetxt(output_path, predictions, delimiter=",", fmt=fmt)
            self._log_info(f"Predictions saved to {output_path}")

            # Save additional info if confidence was requested
            if self.args.show_confidence and "predicted_classes" in output_data:
                # Save classes and confidence alongside
                classes_path = output_path.replace('.csv', '_classes.csv')
                confidence_path = output_path.replace('.csv', '_confidence.csv')
                
                np.savetxt(classes_path, output_data["predicted_classes"].reshape(-1, 1), 
                          delimiter=",", fmt='%d')
                np.savetxt(confidence_path, output_data["confidence"].reshape(-1, 1),
                          delimiter=",", fmt=fmt)
                
                self._log_info(f"Classes saved to {classes_path}")
                self._log_info(f"Confidence saved to {confidence_path}")

        except Exception as e:
            self._log_error(f"Error saving predictions to CSV: {e}")

    def _save_json(self, output_data: dict, input_data: np.ndarray, precision: int):
        """Save predictions to JSON file."""
        output_path = self.args.output_json or self.args.output_csv
        if output_path and not output_path.endswith('.json'):
            output_path = output_path.replace('.csv', '.json') if '.csv' in output_path else output_path + '.json'

        try:
            predictions = output_data["predictions"]
            
            result = {
                "num_samples": len(predictions),
                "predictions": predictions.tolist(),
            }

            if self.args.show_confidence and "predicted_classes" in output_data:
                result["predicted_classes"] = output_data["predicted_classes"].tolist()
                result["confidence"] = output_data["confidence"].tolist()
                
                if "top_k_classes" in output_data:
                    result["top_k_classes"] = output_data["top_k_classes"].tolist()
                    result["top_k_confidence"] = output_data["top_k_confidence"].tolist()

            # Summary statistics
            result["summary"] = {
                "mean": float(np.mean(predictions)),
                "std": float(np.std(predictions)),
                "min": float(np.min(predictions)),
                "max": float(np.max(predictions)),
            }

            if output_path:
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                self._log_info(f"Predictions saved to {output_path}")
            else:
                # Print to stdout
                print(json.dumps(result, indent=2))

        except Exception as e:
            self._log_error(f"Error saving predictions to JSON: {e}")
