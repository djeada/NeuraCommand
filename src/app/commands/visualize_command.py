"""
Visualize command - training progress visualization.

Provides ASCII-based visualization of training progress including
loss curves and metric history for terminal-based environments.
"""

import argparse
import json
from typing import Dict, List, Optional, Any

import numpy as np

from app.commands.base_command import BaseCommand


class VisualizeCommand(BaseCommand):
    """Visualize training progress and metrics."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--history-json",
            type=BaseCommand.file_path,
            required=True,
            help="Path to training history JSON file",
        )
        parser.add_argument(
            "--metric",
            type=str,
            default="loss",
            help="Metric to visualize (default: loss)",
        )
        parser.add_argument(
            "--width",
            type=int,
            default=60,
            help="Width of ASCII chart (default: 60)",
        )
        parser.add_argument(
            "--height",
            type=int,
            default=20,
            help="Height of ASCII chart (default: 20)",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["ascii", "json", "table"],
            default="ascii",
            help="Output format (default: ascii)",
        )

    def execute(self):
        try:
            with open(self.args.history_json, "r") as f:
                history = json.load(f)
        except Exception as e:
            print(f"Error loading history file: {e}")
            return

        if self.args.format == "json":
            self._output_json(history)
        elif self.args.format == "table":
            self._output_table(history)
        else:
            self._output_ascii(history)

    def _output_json(self, history: Dict[str, List[float]]):
        """Output history as formatted JSON."""
        # Add summary statistics
        summary = {}
        for metric, values in history.items():
            if isinstance(values, list) and len(values) > 0:
                summary[metric] = {
                    "values": values,
                    "epochs": len(values),
                    "initial": values[0],
                    "final": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "improvement": values[0] - values[-1] if "loss" in metric else values[-1] - values[0],
                }
        print(json.dumps(summary, indent=2))

    def _output_table(self, history: Dict[str, List[float]]):
        """Output history as a formatted table."""
        # Get all metrics
        metrics = [m for m in history.keys() if isinstance(history[m], list) and len(history[m]) > 0]
        if not metrics:
            print("No metrics found in history")
            return

        epochs = max(len(history[m]) for m in metrics)
        
        # Header
        header = "Epoch"
        for m in metrics:
            header += f" | {m:>12}"
        print(header)
        print("-" * len(header))

        # Data rows
        for i in range(epochs):
            row = f"{i+1:5}"
            for m in metrics:
                if i < len(history[m]):
                    row += f" | {history[m][i]:12.6f}"
                else:
                    row += f" | {'':>12}"
            print(row)

        # Summary
        print("-" * len(header))
        print("\nTraining Summary:")
        for m in metrics:
            values = history[m]
            if len(values) > 0:
                improvement = values[0] - values[-1] if "loss" in m else values[-1] - values[0]
                print(f"  {m}:")
                print(f"    Initial: {values[0]:.6f}")
                print(f"    Final:   {values[-1]:.6f}")
                print(f"    Best:    {min(values) if 'loss' in m else max(values):.6f}")
                if values[0] != 0:
                    print(f"    Change:  {improvement:+.6f} ({abs(improvement)/abs(values[0])*100:.1f}%)")
                else:
                    print(f"    Change:  {improvement:+.6f}")

    def _output_ascii(self, history: Dict[str, List[float]]):
        """Output ASCII chart visualization."""
        metric = self.args.metric
        width = self.args.width
        height = self.args.height

        # Find metrics to plot
        metrics_to_plot = []
        if metric in history:
            metrics_to_plot.append(metric)
        # Also check for validation version
        val_metric = f"val_{metric}"
        if val_metric in history:
            metrics_to_plot.append(val_metric)

        if not metrics_to_plot:
            available = [k for k in history.keys() if isinstance(history[k], list) and len(history[k]) > 0]
            print(f"Metric '{metric}' not found in history.")
            print(f"Available metrics: {', '.join(available)}")
            return

        print("=" * (width + 12))
        print(f"Training Progress Visualization")
        print("=" * (width + 12))

        for m in metrics_to_plot:
            values = history[m]
            if not values:
                continue

            print(f"\n--- {m} ---")
            self._draw_ascii_chart(values, width, height, m)
            
            # Print summary statistics
            print(f"\n  Epochs: {len(values)}")
            print(f"  Initial: {values[0]:.6f}")
            print(f"  Final:   {values[-1]:.6f}")
            print(f"  Min:     {min(values):.6f} (epoch {values.index(min(values)) + 1})")
            print(f"  Max:     {max(values):.6f} (epoch {values.index(max(values)) + 1})")

            if len(values) >= 2:
                improvement = values[0] - values[-1]
                pct = abs(improvement) / abs(values[0]) * 100 if values[0] != 0 else 0
                direction = "decreased" if improvement > 0 else "increased"
                print(f"  Change:  {abs(improvement):.6f} ({direction} by {pct:.1f}%)")

        # If we have both train and val, show gap analysis
        if "loss" in history and "val_loss" in history:
            train_loss = history["loss"]
            val_loss = history["val_loss"]
            min_len = min(len(train_loss), len(val_loss))
            if min_len > 0:
                print("\n--- Train/Val Gap Analysis ---")
                final_gap = val_loss[-1] - train_loss[-1] if min_len > 0 else 0
                print(f"  Final train loss: {train_loss[-1]:.6f}")
                print(f"  Final val loss:   {val_loss[-1]:.6f}")
                print(f"  Gap:              {final_gap:.6f}")
                if final_gap > train_loss[-1] * 0.5:
                    print("  Warning: Large gap suggests potential overfitting")

        print("\n" + "=" * (width + 12))

    def _draw_ascii_chart(self, values: List[float], width: int, height: int, title: str):
        """Draw an ASCII line chart."""
        if not values:
            return

        n_points = len(values)
        min_val = min(values)
        max_val = max(values)
        val_range = max_val - min_val if max_val != min_val else 1

        # Create the chart grid
        chart = [[' ' for _ in range(width)] for _ in range(height)]

        # Scale values to chart dimensions
        def scale_y(v):
            return int((height - 1) * (1 - (v - min_val) / val_range))

        def scale_x(i):
            return int((width - 1) * i / (n_points - 1)) if n_points > 1 else 0

        # Plot points and connect with lines
        prev_x, prev_y = None, None
        for i, v in enumerate(values):
            x = scale_x(i)
            y = scale_y(v)
            y = max(0, min(height - 1, y))

            # Draw connecting line if not first point
            if prev_x is not None and prev_y is not None:
                # Simple line drawing
                if prev_x != x:
                    steps = abs(x - prev_x)
                    for step in range(1, steps):
                        inter_x = prev_x + step
                        inter_y = prev_y + int((y - prev_y) * step / steps)
                        inter_y = max(0, min(height - 1, inter_y))
                        if 0 <= inter_x < width:
                            chart[inter_y][inter_x] = '·'

            # Plot the point
            if 0 <= x < width and 0 <= y < height:
                chart[y][x] = '●'

            prev_x, prev_y = x, y

        # Draw y-axis labels
        y_labels = []
        for i in range(height):
            if i == 0:
                y_labels.append(f"{max_val:8.4f}")
            elif i == height - 1:
                y_labels.append(f"{min_val:8.4f}")
            elif i == height // 2:
                mid_val = (max_val + min_val) / 2
                y_labels.append(f"{mid_val:8.4f}")
            else:
                y_labels.append("        ")

        # Print the chart
        print("         │" + "─" * width + "┐")
        for i, row in enumerate(chart):
            label = y_labels[i] if i < len(y_labels) else "        "
            print(f"{label} │{''.join(row)}│")
        print("         └" + "─" * width + "┘")

        # X-axis labels
        x_axis = "         "
        if n_points > 1:
            x_axis += f"1{' ' * (width // 2 - 1)}epoch{' ' * (width // 2 - 5)}{n_points}"
        else:
            x_axis += f"epoch 1{' ' * (width - 7)}"
        print(x_axis)


def save_training_history(history: Dict[str, List[float]], path: str):
    """Save training history to a JSON file for later visualization.
    
    This is a utility function that can be used programmatically to save
    the history returned by model.train() for later analysis.
    
    Args:
        history: Dictionary of metric names to lists of values
        path: Path to save the JSON file
        
    Example:
        history = model.train(X, y, epochs=100)
        save_training_history(history, 'training_history.json')
    """
    with open(path, "w") as f:
        json.dump(history, f, indent=2)
