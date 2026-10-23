"""Export lean inference artifacts or standalone embedded C99 networks."""

import argparse
import sys

from app.commands.base_command import BaseCommand


class ExportCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model",
            required=True,
            help="Path to a modern training model (base path).",
        )
        parser.add_argument("--output", required=True, help="Output base path.")
        parser.add_argument(
            "--quantize",
            choices=["none", "float16", "int8"],
            default="none",
            help="Weight storage precision (default: none / float32).",
        )
        parser.add_argument(
            "--format",
            choices=["inference", "c"],
            default="inference",
            help="JSON+NPZ inference artifact or standalone C99 (default: inference).",
        )
        parser.add_argument(
            "--per-channel",
            action="store_true",
            help="Use one INT8 scale per Dense output channel for better accuracy.",
        )
        parser.add_argument(
            "--prune-threshold",
            type=float,
            default=0.0,
            help="Remove hidden channels with outgoing weights at/below this value; "
            "nonzero values are lossy and require accuracy validation (default: 0).",
        )
        parser.add_argument(
            "--no-optimize",
            action="store_true",
            help="Disable BatchNorm folding, identity removal, and channel compaction.",
        )
        parser.add_argument(
            "--max-weight-bytes",
            type=int,
            default=None,
            help="Reject export if parameter storage exceeds this byte budget.",
        )
        parser.add_argument(
            "--max-scratch-bytes",
            type=int,
            default=None,
            help="Reject C export if its scratch RAM exceeds this byte budget.",
        )
        parser.add_argument(
            "--name",
            default="neura_model",
            help="C symbol prefix (default: neura_model).",
        )

    def execute(self):
        from neura_command.serialization import load_model
        from neura_command.inference import InferenceModel

        try:
            output_format = getattr(self.args, "format", "inference")
            max_weights = getattr(self.args, "max_weight_bytes", None)
            max_scratch = getattr(self.args, "max_scratch_bytes", None)
            if any(
                value is not None and value < 0 for value in (max_weights, max_scratch)
            ):
                raise ValueError("Memory budgets must be non-negative byte counts")
            if output_format != "c" and max_scratch is not None:
                raise ValueError(
                    "--max-scratch-bytes requires --format c; NumPy temporary RAM is not statically bounded"
                )
            print(f"Loading model from '{self.args.model}'...")
            model = load_model(self.args.model)
            print(f"Original model: {model.num_parameters:,} parameters")
            quantize = self.args.quantize if self.args.quantize != "none" else None
            inference = InferenceModel.from_sequential(
                model,
                quantize=quantize,
                optimize=not getattr(self.args, "no_optimize", False),
                per_channel=getattr(self.args, "per_channel", False),
                prune_threshold=getattr(self.args, "prune_threshold", 0.0),
            )
            if output_format == "c":
                report = inference.export_c(
                    self.args.output,
                    name=getattr(self.args, "name", "neura_model"),
                    max_weight_bytes=max_weights,
                    max_scratch_bytes=max_scratch,
                )
                print(
                    f"Exported C99: {report['source_path']} and {report['header_path']}"
                )
                print(
                    f"Weights: {report['weight_bytes']} bytes; scratch RAM: {report['scratch_bytes']} bytes"
                )
                print(f"Resource report: {report['report_path']}")
            else:
                if max_weights is not None and inference.memory_bytes > max_weights:
                    raise ValueError(
                        f"Weight budget exceeded: {inference.memory_bytes} bytes required, limit {max_weights}"
                    )
                inference.save(self.args.output)
                json_path, npz_path = InferenceModel.artifact_paths(self.args.output)
                print(f"Exported inference artifact to '{json_path}' and '{npz_path}'")
                print(f"Inference parameter storage: {inference.memory_bytes} bytes")
            print(
                f"Exported layers: {len(inference.layers)}; quantization: {quantize or 'float32'}"
            )
        except Exception as error:
            print(f"Error exporting model: {error}", file=sys.stderr)
            raise SystemExit(1) from error
