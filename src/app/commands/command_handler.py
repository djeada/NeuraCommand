import argparse

from app.commands.create_network_command import CreateNetworkCommand
from app.commands.predict_command import PredictCommand
from app.commands.train_network_command import TrainNetworkCommand
from app.commands.info_command import InfoCommand
from app.commands.export_command import ExportCommand

try:
    from neura_command import __version__
except ImportError:
    __version__ = "0.1.0"


class CommandHandler:
    def __init__(self):
        self.command_map = {
            "create": CreateNetworkCommand,
            "train": TrainNetworkCommand,
            "predict": PredictCommand,
            "info": InfoCommand,
            "export": ExportCommand,
        }
        self.parser = self._create_parser()
        self.args, unknown_args = self.parser.parse_known_args()
        
        if self.args.version:
            print(f"neura_command {__version__}")
            exit(0)
        
        self._handle_arguments(unknown_args)

    def _create_parser(self):
        parser = argparse.ArgumentParser(description="Neural Network Command Line Tool")
        parser.add_argument(
            "--version",
            action="store_true",
            help="Show version number and exit",
        )
        parser.add_argument(
            "--action",
            type=str,
            choices=self.command_map.keys(),
            required=False,
            help="The action to perform (create, train, predict, info, export)",
        )
        return parser

    def _handle_arguments(self, unknown_args):
        if not self.args.action:
            self.parser.error("--action is required unless --version is specified")
        command_class = self.command_map.get(self.args.action)
        if command_class:
            # Create a new subparser for the command
            subparser = argparse.ArgumentParser()
            command_class.add_arguments(subparser)
            self.command_args = subparser.parse_args(unknown_args)
        else:
            raise ValueError(f"Unknown command: {self.args.action}")

    def execute_command(self):
        command = self.command_map.get(self.args.action)
        if command:
            command_instance = command(self.command_args)
            command_instance.execute()
        else:
            print("No valid command provided.")
