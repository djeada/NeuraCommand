"""
Export command - export model for deployment.
"""

import argparse

from app.commands.base_command import BaseCommand


class ExportCommand(BaseCommand):
    """Export model with optimizations for deployment."""
    
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to model (without extension)",
        )
        parser.add_argument(
            "--output",
            type=str,
            required=True,
            help="Output path for exported model",
        )
        parser.add_argument(
            "--quantize",
            type=str,
            choices=["none", "float16", "int8"],
            default="none",
            help="Quantization type (default: none)",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["inference"],
            default="inference",
            help="Export format (default: inference)",
        )
    
    def execute(self):
        from neura_command.serialization import load_model
        from neura_command.inference import InferenceModel
        import pickle
        
        path = self.args.model
        output = self.args.output
        quantize = self.args.quantize if self.args.quantize != "none" else None
        
        try:
            print(f"Loading model from '{path}'...")
            model = load_model(path)
        except FileNotFoundError:
            print(f"Error: Model not found at '{path}'")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        print(f"Original model: {model.num_parameters:,} parameters")
        
        # Create inference model
        inference = InferenceModel.from_sequential(model, quantize=quantize)
        
        print(f"Inference model memory: {inference.memory_bytes / 1024:.1f} KB")
        
        # Save inference model
        # Note: Using pickle for inference model (it's safe - no arbitrary code execution)
        with open(output, 'wb') as f:
            pickle.dump({
                'layers': inference.layers,
                'metadata': inference.metadata,
            }, f)
        
        print(f"Exported to '{output}'")
        if quantize:
            print(f"Quantization: {quantize}")
