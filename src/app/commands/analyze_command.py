"""
Analyze command - detailed network analytics and diagnostics.

Provides comprehensive metrics, error analysis, and performance diagnostics
for trained neural network models.
"""

import argparse
import json
from typing import Dict, Any, Optional, List

import numpy as np

from app.commands.base_command import BaseCommand


class AnalyzeCommand(BaseCommand):
    """Analyze a neural network model with detailed metrics and diagnostics."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to model (without extension)",
        )
        parser.add_argument(
            "--features-csv",
            type=BaseCommand.file_path,
            required=False,
            help="Path to CSV file with input features for analysis",
        )
        parser.add_argument(
            "--targets-csv",
            type=BaseCommand.file_path,
            required=False,
            help="Path to CSV file with target values for error analysis",
        )
        parser.add_argument(
            "--analysis-type",
            type=str,
            choices=["architecture", "weights", "performance", "errors", "all"],
            default="all",
            help="Type of analysis to perform (default: all)",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)",
        )
        parser.add_argument(
            "--task-type",
            type=str,
            choices=["regression", "classification"],
            default="regression",
            help="Task type for appropriate metrics (default: regression)",
        )

    def execute(self):
        from neura_command.serialization import load_model, get_model_info

        path = self.args.model
        analysis_type = self.args.analysis_type

        try:
            model = load_model(path)
            info = get_model_info(path)
        except FileNotFoundError:
            print(f"Error: Model not found at '{path}'")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        results: Dict[str, Any] = {}

        # Architecture analysis
        if analysis_type in ["architecture", "all"]:
            results["architecture"] = self._analyze_architecture(model, info)

        # Weight statistics analysis
        if analysis_type in ["weights", "all"]:
            results["weights"] = self._analyze_weights(model)

        # Performance analysis (requires data)
        if analysis_type in ["performance", "all"] and self.args.features_csv:
            features = self._load_csv(self.args.features_csv)
            targets = self._load_csv(self.args.targets_csv) if self.args.targets_csv else None
            if features is not None:
                results["performance"] = self._analyze_performance(
                    model, features, targets, self.args.task_type
                )

        # Error analysis (requires both features and targets)
        if analysis_type in ["errors", "all"] and self.args.features_csv and self.args.targets_csv:
            features = self._load_csv(self.args.features_csv)
            targets = self._load_csv(self.args.targets_csv)
            if features is not None and targets is not None:
                results["errors"] = self._analyze_errors(
                    model, features, targets, self.args.task_type
                )

        # Output results
        if self.args.format == "json":
            print(json.dumps(results, indent=2, default=self._json_serializer))
        else:
            self._print_text_results(results)

    def _load_csv(self, path: str) -> Optional[np.ndarray]:
        """Load CSV file as numpy array."""
        try:
            return np.loadtxt(path, delimiter=",")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def _analyze_architecture(self, model, info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model architecture in detail."""
        layer_details = []
        total_params = 0
        total_memory = 0

        for i, layer in enumerate(model.layers):
            params = layer.num_parameters
            total_params += params
            
            # Estimate memory (4 bytes per float32 parameter)
            memory = params * 4
            total_memory += memory

            layer_info = {
                "index": i,
                "type": layer.__class__.__name__,
                "parameters": params,
                "memory_bytes": memory,
            }

            # Add layer-specific details
            if hasattr(layer, "in_features"):
                layer_info["in_features"] = layer.in_features
            if hasattr(layer, "out_features"):
                layer_info["out_features"] = layer.out_features
            if hasattr(layer, "rate"):
                layer_info["dropout_rate"] = layer.rate
            if hasattr(layer, "use_bias"):
                layer_info["use_bias"] = layer.use_bias

            layer_details.append(layer_info)

        return {
            "num_layers": len(model.layers),
            "total_parameters": total_params,
            "total_memory_bytes": total_memory,
            "total_memory_kb": total_memory / 1024,
            "optimizer": model.optimizer.__class__.__name__,
            "loss_function": model.loss.__class__.__name__,
            "layers": layer_details,
            "saved_at": info.get("saved_at"),
            "neura_command_version": info.get("neura_command_version"),
        }

    def _analyze_weights(self, model) -> Dict[str, Any]:
        """Analyze weight statistics for each layer."""
        layer_stats = []

        for i, layer in enumerate(model.layers):
            params = layer.parameters()
            if not params:
                continue

            for j, param in enumerate(params):
                stats = {
                    "layer_index": i,
                    "layer_type": layer.__class__.__name__,
                    "param_index": j,
                    "shape": list(param.shape),
                    "size": param.size,
                    "mean": float(np.mean(param)),
                    "std": float(np.std(param)),
                    "min": float(np.min(param)),
                    "max": float(np.max(param)),
                    "abs_mean": float(np.mean(np.abs(param))),
                    "zeros_ratio": float(np.sum(param == 0) / param.size),
                    "near_zero_ratio": float(np.sum(np.abs(param) < 1e-6) / param.size),
                }
                layer_stats.append(stats)

        # Global statistics
        all_params = np.concatenate([p.flatten() for layer in model.layers for p in layer.parameters()])
        global_stats = {
            "total_parameters": len(all_params),
            "global_mean": float(np.mean(all_params)) if len(all_params) > 0 else 0,
            "global_std": float(np.std(all_params)) if len(all_params) > 0 else 0,
            "global_min": float(np.min(all_params)) if len(all_params) > 0 else 0,
            "global_max": float(np.max(all_params)) if len(all_params) > 0 else 0,
        }

        return {
            "global": global_stats,
            "layers": layer_stats,
        }

    def _analyze_performance(
        self, model, features: np.ndarray, targets: Optional[np.ndarray], task_type: str
    ) -> Dict[str, Any]:
        """Analyze model performance on given data."""
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)

        predictions = model.predict(features)

        results = {
            "num_samples": len(features),
            "input_shape": list(features.shape),
            "output_shape": list(predictions.shape),
        }

        if targets is not None:
            # Ensure targets are 2D
            if targets.ndim == 1:
                targets = targets.reshape(-1, 1)

            if task_type == "regression":
                results.update(self._regression_metrics(targets, predictions))
            else:
                results.update(self._classification_metrics(targets, predictions))

        # Prediction statistics
        results["prediction_stats"] = {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
        }

        return results

    def _regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate regression metrics."""
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-10)))

        # Mean Absolute Percentage Error (MAPE)
        non_zero_mask = np.abs(y_true) > 1e-10
        if np.any(non_zero_mask):
            mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
        else:
            mape = float("inf")

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "mape": mape,
        }

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate classification metrics."""
        # Convert predictions to class labels
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            pred_labels = np.argmax(y_pred, axis=1)
        else:
            pred_labels = (y_pred.flatten() > 0.5).astype(int)

        if y_true.ndim > 1 and y_true.shape[1] > 1:
            true_labels = np.argmax(y_true, axis=1)
        else:
            true_labels = y_true.flatten().astype(int)

        # Accuracy
        accuracy = float(np.mean(pred_labels == true_labels))

        # Get unique classes
        classes = np.unique(np.concatenate([true_labels, pred_labels]))
        num_classes = len(classes)

        metrics = {
            "accuracy": accuracy,
            "num_classes": num_classes,
        }

        # Per-class metrics for binary or multi-class
        if num_classes <= 10:  # Only compute detailed metrics for reasonable number of classes
            confusion = self._confusion_matrix(true_labels, pred_labels, classes)
            metrics["confusion_matrix"] = confusion.tolist()

            # Precision, Recall, F1 per class
            class_metrics = []
            for i, c in enumerate(classes):
                tp = confusion[i, i]
                fp = np.sum(confusion[:, i]) - tp
                fn = np.sum(confusion[i, :]) - tp

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)

                class_metrics.append({
                    "class": int(c),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "support": int(np.sum(true_labels == c)),
                })

            metrics["class_metrics"] = class_metrics

            # Macro averages
            metrics["macro_precision"] = float(np.mean([m["precision"] for m in class_metrics]))
            metrics["macro_recall"] = float(np.mean([m["recall"] for m in class_metrics]))
            metrics["macro_f1"] = float(np.mean([m["f1_score"] for m in class_metrics]))

        return metrics

    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        n_classes = len(classes)
        matrix = np.zeros((n_classes, n_classes), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        for t, p in zip(y_true, y_pred):
            if t in class_to_idx and p in class_to_idx:
                matrix[class_to_idx[t], class_to_idx[p]] += 1
        
        return matrix

    def _analyze_errors(
        self, model, features: np.ndarray, targets: np.ndarray, task_type: str
    ) -> Dict[str, Any]:
        """Analyze prediction errors in detail."""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        predictions = model.predict(features)
        errors = predictions - targets

        results = {
            "num_samples": len(features),
        }

        if task_type == "regression":
            # Error distribution
            abs_errors = np.abs(errors)
            results["error_distribution"] = {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
                "median": float(np.median(errors)),
                "abs_mean": float(np.mean(abs_errors)),
                "abs_median": float(np.median(abs_errors)),
            }

            # Percentiles
            percentiles = [5, 25, 50, 75, 95]
            results["error_percentiles"] = {
                f"p{p}": float(np.percentile(abs_errors, p)) for p in percentiles
            }

            # Identify worst predictions
            worst_indices = np.argsort(abs_errors.flatten())[-5:][::-1]
            results["worst_predictions"] = [
                {
                    "index": int(idx),
                    "predicted": float(predictions.flatten()[idx]),
                    "actual": float(targets.flatten()[idx]),
                    "error": float(errors.flatten()[idx]),
                    "abs_error": float(abs_errors.flatten()[idx]),
                }
                for idx in worst_indices
            ]

            # Error by target range
            target_flat = targets.flatten()
            if len(np.unique(target_flat)) > 1:
                quartiles = np.percentile(target_flat, [25, 50, 75])
                ranges = [
                    ("Q1 (0-25%)", target_flat <= quartiles[0]),
                    ("Q2 (25-50%)", (target_flat > quartiles[0]) & (target_flat <= quartiles[1])),
                    ("Q3 (50-75%)", (target_flat > quartiles[1]) & (target_flat <= quartiles[2])),
                    ("Q4 (75-100%)", target_flat > quartiles[2]),
                ]
                
                results["error_by_target_range"] = {
                    name: {
                        "count": int(np.sum(mask)),
                        "mean_abs_error": float(np.mean(abs_errors.flatten()[mask])) if np.any(mask) else 0,
                    }
                    for name, mask in ranges
                }
        else:
            # Classification error analysis
            if predictions.ndim > 1 and predictions.shape[1] > 1:
                pred_labels = np.argmax(predictions, axis=1)
                pred_confidence = np.max(predictions, axis=1)
            else:
                pred_labels = (predictions.flatten() > 0.5).astype(int)
                pred_confidence = np.abs(predictions.flatten() - 0.5) * 2

            if targets.ndim > 1 and targets.shape[1] > 1:
                true_labels = np.argmax(targets, axis=1)
            else:
                true_labels = targets.flatten().astype(int)

            # Misclassification analysis
            incorrect_mask = pred_labels != true_labels
            results["misclassification"] = {
                "count": int(np.sum(incorrect_mask)),
                "rate": float(np.mean(incorrect_mask)),
            }

            if np.any(incorrect_mask):
                results["misclassification"]["avg_confidence"] = float(np.mean(pred_confidence[incorrect_mask]))
                results["misclassification"]["min_confidence"] = float(np.min(pred_confidence[incorrect_mask]))
                results["misclassification"]["max_confidence"] = float(np.max(pred_confidence[incorrect_mask]))

            # Confidence analysis
            results["confidence_stats"] = {
                "mean": float(np.mean(pred_confidence)),
                "std": float(np.std(pred_confidence)),
                "min": float(np.min(pred_confidence)),
                "max": float(np.max(pred_confidence)),
            }

        return results

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _print_text_results(self, results: Dict[str, Any]):
        """Print results in human-readable format."""
        print("=" * 60)
        print("Network Analytics Report")
        print("=" * 60)

        if "architecture" in results:
            arch = results["architecture"]
            print("\n--- Architecture Analysis ---")
            print(f"Total layers: {arch['num_layers']}")
            print(f"Total parameters: {arch['total_parameters']:,}")
            print(f"Memory usage: {arch['total_memory_kb']:.2f} KB")
            print(f"Optimizer: {arch['optimizer']}")
            print(f"Loss function: {arch['loss_function']}")
            print(f"Saved at: {arch.get('saved_at', 'unknown')}")
            
            print("\nLayer details:")
            for layer in arch["layers"]:
                params_str = f" ({layer['parameters']:,} params)" if layer['parameters'] > 0 else ""
                extra = ""
                if "in_features" in layer:
                    extra = f" [{layer['in_features']} -> {layer['out_features']}]"
                print(f"  {layer['index']}: {layer['type']}{extra}{params_str}")

        if "weights" in results:
            weights = results["weights"]
            print("\n--- Weight Statistics ---")
            g = weights["global"]
            print(f"Total parameters: {g['total_parameters']:,}")
            print(f"Global mean: {g['global_mean']:.6f}")
            print(f"Global std: {g['global_std']:.6f}")
            print(f"Range: [{g['global_min']:.6f}, {g['global_max']:.6f}]")
            
            print("\nPer-layer statistics:")
            for stat in weights["layers"]:
                print(f"  Layer {stat['layer_index']} ({stat['layer_type']}) param {stat['param_index']}:")
                print(f"    Shape: {stat['shape']}, Mean: {stat['mean']:.4f}, Std: {stat['std']:.4f}")
                print(f"    Near-zero ratio: {stat['near_zero_ratio']:.2%}")

        if "performance" in results:
            perf = results["performance"]
            print("\n--- Performance Analysis ---")
            print(f"Samples evaluated: {perf['num_samples']}")
            print(f"Input shape: {perf['input_shape']}")
            print(f"Output shape: {perf['output_shape']}")

            if "mse" in perf:  # Regression metrics
                print(f"\nRegression Metrics:")
                print(f"  MSE: {perf['mse']:.6f}")
                print(f"  RMSE: {perf['rmse']:.6f}")
                print(f"  MAE: {perf['mae']:.6f}")
                print(f"  R² Score: {perf['r2_score']:.4f}")
                if perf['mape'] != float('inf'):
                    print(f"  MAPE: {perf['mape']:.2f}%")

            if "accuracy" in perf:  # Classification metrics
                print(f"\nClassification Metrics:")
                print(f"  Accuracy: {perf['accuracy']:.4f} ({perf['accuracy']*100:.2f}%)")
                if "macro_precision" in perf:
                    print(f"  Macro Precision: {perf['macro_precision']:.4f}")
                    print(f"  Macro Recall: {perf['macro_recall']:.4f}")
                    print(f"  Macro F1: {perf['macro_f1']:.4f}")

                if "confusion_matrix" in perf:
                    print("\n  Confusion Matrix:")
                    matrix = np.array(perf["confusion_matrix"])
                    print("  " + str(matrix).replace("\n", "\n  "))

                if "class_metrics" in perf:
                    print("\n  Per-class metrics:")
                    for cm in perf["class_metrics"]:
                        print(f"    Class {cm['class']}: P={cm['precision']:.3f} R={cm['recall']:.3f} F1={cm['f1_score']:.3f} (n={cm['support']})")

            if "prediction_stats" in perf:
                ps = perf["prediction_stats"]
                print(f"\nPrediction Statistics:")
                print(f"  Mean: {ps['mean']:.6f}, Std: {ps['std']:.6f}")
                print(f"  Range: [{ps['min']:.6f}, {ps['max']:.6f}]")

        if "errors" in results:
            errors = results["errors"]
            print("\n--- Error Analysis ---")
            print(f"Samples analyzed: {errors['num_samples']}")

            if "error_distribution" in errors:
                ed = errors["error_distribution"]
                print(f"\nError Distribution:")
                print(f"  Mean error: {ed['mean']:.6f}")
                print(f"  Std: {ed['std']:.6f}")
                print(f"  Median: {ed['median']:.6f}")
                print(f"  Range: [{ed['min']:.6f}, {ed['max']:.6f}]")
                print(f"  Mean absolute error: {ed['abs_mean']:.6f}")

            if "error_percentiles" in errors:
                print("\nError Percentiles (absolute):")
                for p, v in errors["error_percentiles"].items():
                    print(f"  {p}: {v:.6f}")

            if "worst_predictions" in errors:
                print("\nWorst Predictions:")
                for wp in errors["worst_predictions"]:
                    print(f"  Sample {wp['index']}: predicted={wp['predicted']:.4f}, actual={wp['actual']:.4f}, error={wp['error']:.4f}")

            if "error_by_target_range" in errors:
                print("\nError by Target Range:")
                for name, data in errors["error_by_target_range"].items():
                    print(f"  {name}: n={data['count']}, MAE={data['mean_abs_error']:.4f}")

            if "misclassification" in errors:
                mc = errors["misclassification"]
                print(f"\nMisclassification Analysis:")
                print(f"  Count: {mc['count']}")
                print(f"  Rate: {mc['rate']:.4f} ({mc['rate']*100:.2f}%)")
                if "avg_confidence" in mc:
                    print(f"  Avg confidence on errors: {mc['avg_confidence']:.4f}")

            if "confidence_stats" in errors:
                cs = errors["confidence_stats"]
                print(f"\nConfidence Statistics:")
                print(f"  Mean: {cs['mean']:.4f}, Std: {cs['std']:.4f}")
                print(f"  Range: [{cs['min']:.4f}, {cs['max']:.4f}]")

        print("\n" + "=" * 60)
