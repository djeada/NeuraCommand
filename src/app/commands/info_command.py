"""
Info command - display model information.
"""

import argparse

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
            model = load_model(path)
        except FileNotFoundError:
            print(f"Error: Model not found at '{path}'")
            print("Expected files: {path}.json and {path}.npz")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        if self.args.format == "json":
            # Add parameter count to info
            info['total_parameters'] = model.num_parameters
            print(json.dumps(info, indent=2, default=str))
        else:
            print(model.summary())
            print(f"\nModel file: {path}")
            print(f"Weights size: {info['weights_size_bytes'] / 1024:.1f} KB")
            if info.get('metadata'):
                print(f"Saved at: {info.get('metadata', {}).get('saved_at', 'unknown')}")
