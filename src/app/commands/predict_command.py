import logging

import numpy as np

from app.commands.base_command import BaseCommand
from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
    FeedForwardNetworkIO,
)


class PredictCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model-pkl",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the neural network file (.pkl).",
        )
        parser.add_argument(
            "--input-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the input CSV file. The number of columns must match the number of inputs required by the .pkl model.",
        )
        parser.add_argument(
            "--output-mode",
            type=str,
            choices=["display", "save"],
            default="display",
            help="Choose 'display' to show predictions on stdout or 'save' to save them to a CSV file.",
        )
        parser.add_argument(
            "--output-csv",
            type=str,
            help="File path to save the prediction results as a CSV. Required if --output-mode is 'save'.",
        )

    def execute(self):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        # Load the model
        try:
            model = FeedForwardNetworkIO.load_from_pickle(self.args.model_pkl)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return

        # Load input data
        try:
            input_data = np.loadtxt(self.args.input_csv, delimiter=",")
            logging.info("Input data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading input data: {e}")
            return

        # Make predictions
        try:
            predictions = model.predict(input_data)
            logging.info("Predictions made successfully.")
        except Exception as e:
            logging.error(f"Error making predictions: {e}")
            return

        # Output handling
        if self.args.output_mode == "display":
            print("Predictions:\n", predictions)
        elif self.args.output_mode == "save":
            if not self.args.output_csv:
                logging.error("Output CSV path is required for 'save' mode.")
                return
            try:
                np.savetxt(self.args.output_csv, predictions, delimiter=",")
                logging.info(f"Predictions saved to {self.args.output_csv}")
            except Exception as e:
                logging.error(f"Error saving predictions to CSV: {e}")
