import json
import logging
from app.commands.base_command import BaseCommand
from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
    FeedForwardNetworkIO,
)
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class CreateNetworkCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--network-architecture-json",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the JSON file describing the neural network's architecture.",
        )
        parser.add_argument(
            "--model-pkl",
            type=str,
            default="network.pkl",
            help="File path to save the serialized network object (.pkl format). Defaults to 'network.pkl'.",
        )

    def execute(self):
        # Set up basic logging configuration
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Load network architecture from JSON file
        try:
            with open(self.args.network_architecture_json, "r") as file:
                logging.info("Loading network architecture from JSON file.")
                data = json.load(file)
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            return

        # Build the network model
        try:
            logging.info("Building the network model.")
            network_architecture = NetworkBuilder.from_json(json_config=data)
            model = network_architecture.build()
        except Exception as e:
            logging.error(f"Error building network model: {e}")
            return

        # Save the model to a .pkl file
        try:
            logging.info("Saving the network model to a .pkl file.")
            FeedForwardNetworkIO.save_to_pickle(model, self.args.model_pkl)
            logging.info(f"Network model successfully saved to {self.args.model_pkl}")
        except Exception as e:
            logging.error(f"Error saving model to file: {e}")
