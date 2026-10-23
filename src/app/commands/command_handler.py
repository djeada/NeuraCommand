"""
Command handler for the NeuraCommand CLI.

Manages parsing and execution of CLI commands using argparse subparsers
for a standard, user-friendly command-line experience.
"""

import argparse
import sys

from app.commands.create_network_command import CreateNetworkCommand
from app.commands.predict_command import PredictCommand
from app.commands.train_network_command import TrainNetworkCommand
from app.commands.info_command import InfoCommand
from app.commands.export_command import ExportCommand
from app.commands.analyze_command import AnalyzeCommand
from app.commands.visualize_command import VisualizeCommand

try:
    from neura_command import __version__
except ImportError:
    __version__ = "0.1.0"


class CommandHandler:
    """Handles CLI command parsing and execution."""

    # Command descriptions for help text
    COMMAND_DESCRIPTIONS = {
        "create": "Create a neural network from JSON architecture (supports modern and legacy formats)",
        "train": "Train a neural network with configurable epochs, batch size, and learning rate",
        "predict": "Make predictions with batch processing, confidence scores, and multiple output formats",
        "info": "Display detailed information about a saved model",
        "export": "Export model for deployment with optional quantization",
        "analyze": "Analyze model architecture, weights, performance metrics, and errors",
        "visualize": "Visualize training progress with ASCII charts from history file",
    }

    def __init__(self):
        self.command_map = {
            "create": CreateNetworkCommand,
            "train": TrainNetworkCommand,
            "predict": PredictCommand,
            "info": InfoCommand,
            "export": ExportCommand,
            "analyze": AnalyzeCommand,
            "visualize": VisualizeCommand,
        }
        self.parser = self._create_parser()
        self.args = self._parse_arguments()

    def _create_parser(self):
        """Create the main argument parser with subparsers for each command."""
        parser = argparse.ArgumentParser(
            prog="neura_command",
            description="NeuraCommand - Neural Network Command Line Tool",
            epilog="Use '%(prog)s <command> --help' for more information on a specific command.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument(
            "--version",
            "-v",
            action="version",
            version=f"neura_command {__version__}",
            help="Show version number and exit",
        )

        # Create subparsers for each command
        subparsers = parser.add_subparsers(
            title="commands",
            dest="command",
            metavar="<command>",
            help="Available commands",
        )

        # Register each command with its subparser
        for name, command_class in self.command_map.items():
            description = self.COMMAND_DESCRIPTIONS.get(name, "")
            subparser = subparsers.add_parser(
                name,
                help=description,
                description=description,
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            command_class.add_arguments(subparser)

        # Also support legacy --action syntax for backward compatibility
        parser.add_argument(
            "--action",
            type=str,
            choices=self.command_map.keys(),
            dest="legacy_action",
            help=argparse.SUPPRESS,  # Hidden from help for cleaner output
        )

        return parser

    def _parse_arguments(self):
        """Parse command line arguments with support for both new and legacy syntax."""
        # First, parse known args to check for legacy --action
        args, remaining = self.parser.parse_known_args()

        # Handle legacy --action syntax
        if args.legacy_action and not getattr(args, 'command', None):
            args.command = args.legacy_action
            # Re-parse with the subcommand's arguments
            command_class = self.command_map.get(args.command)
            if command_class:
                subparser = argparse.ArgumentParser(
                    prog=f"neura_command --action {args.command}",
                    description=self.COMMAND_DESCRIPTIONS.get(args.command, ""),
                )
                command_class.add_arguments(subparser)
                subargs = subparser.parse_args(remaining)
                # Merge subargs into args
                for key, value in vars(subargs).items():
                    setattr(args, key, value)
        elif remaining:
            self.parser.error("unrecognized arguments: " + " ".join(remaining))

        return args

    def execute_command(self):
        """Execute the parsed command."""
        command = getattr(self.args, 'command', None)
        if not command:
            self.parser.print_help()
            print("\nError: Please specify a command.")
            print("Use 'neura_command <command> --help' for more information on a command.")
            sys.exit(1)

        command_class = self.command_map.get(command)
        if command_class:
            command_instance = command_class(self.args)
            command_instance.execute()
        else:
            print(f"Error: Unknown command '{command}'")
            self.parser.print_help()
            sys.exit(1)
