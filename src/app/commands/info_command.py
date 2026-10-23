"""
Info command - display model information.
"""

import argparse
import sys

from app.commands.base_command import BaseCommand


class InfoCommand(BaseCommand):
    """Display information about a saved model."""

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to model (without extension)",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["text", "json"],
            default="text",
            help="Output format (default: text)",
        )

    def execute(self):
        from neura_command.serialization import get_model_info, load_model
        import json

        path = self.args.model

        try:
            info = get_model_info(path)
            if info["artifact_type"] == "inference_model":
                from neura_command.inference import InferenceModel

                model = InferenceModel.load(path)
                parameter_bytes = model.memory_bytes
                summary = model.info()
            else:
                model = load_model(path)
                parameter_bytes = sum(value.nbytes for value in model._weight_arrays())
                summary = model.summary()
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            raise SystemExit(1) from e

        info["total_parameters"] = model.num_parameters
        info["parameter_bytes"] = parameter_bytes
        info["device"] = model.device

        if self.args.format == "json":
            # Add parameter count to info
            print(json.dumps(info, indent=2, default=str))
        else:
            print(summary)
            print(f"\nModel file: {path}")
            print(f"Weights size: {info['weights_size_bytes'] / 1024:.1f} KB")
            if info.get("saved_at"):
                print(f"Saved at: {info['saved_at']}")
