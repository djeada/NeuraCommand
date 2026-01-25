import json
import logging

import numpy as np

from app.commands.base_command import BaseCommand
from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
    FeedForwardNetworkIO,
)


class TrainNetworkCommand(BaseCommand):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--model-pkl",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the neural network file (.pkl).",
        )
        parser.add_argument(
            "--features-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the CSV file containing the input features for training the model.",
        )
        parser.add_argument(
            "--targets-csv",
            type=BaseCommand.file_path,
            required=True,
            help="Path to the CSV file containing the target values corresponding to the input features.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of epochs to train the network. An epoch is one complete presentation of the data set to be learned.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Flag to indicate whether the trained model should overwrite the existing .pkl file.",
        )
        parser.add_argument(
            "--output-model-pkl",
            type=str,
            help="Custom path to save the trained network model file (.pkl). Meaningful if --overwrite is set to False.",
        )
        parser.add_argument(
            "--save-history",
            type=str,
            help="Path to save training history as JSON file for later visualization.",
        )
        parser.add_argument(
            "--show-summary",
            action="store_true",
            help="Display training summary after training completes.",
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

        # Load training data
        try:
            features = np.loadtxt(self.args.features_csv, delimiter=",")
            targets = np.loadtxt(self.args.targets_csv, delimiter=",")
            logging.info("Training data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            return

        # Train the model
        try:
            history = model.train(features, targets, epochs=self.args.epochs)
            logging.info(f"Model trained successfully for {self.args.epochs} epochs.")
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return

        # Save training history if requested
        if hasattr(self.args, 'save_history') and self.args.save_history:
            try:
                # Handle both dict and None returns from train()
                if history is None:
                    history = {"epochs": self.args.epochs}
                with open(self.args.save_history, "w") as f:
                    json.dump(history, f, indent=2)
                logging.info(f"Training history saved to {self.args.save_history}")
            except Exception as e:
                logging.warning(f"Could not save training history: {e}")

        # Show training summary if requested
        if hasattr(self.args, 'show_summary') and self.args.show_summary:
            self._display_training_summary(history)

        # Save the model
        output_path = (
            self.args.model_pkl if self.args.overwrite else self.args.output_model_pkl
        )
        if not output_path:
            logging.error("Output model path is not specified.")
            return
        try:
            FeedForwardNetworkIO.save_to_pickle(model, output_path)
            logging.info(f"Trained model saved to {output_path}")
        except Exception as e:
            logging.error(f"Error saving the trained model: {e}")

    def _display_training_summary(self, history):
        """Display a summary of training progress."""
        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)

        if history is None:
            print(f"Epochs completed: {self.args.epochs}")
            print("(Detailed history not available for this model type)")
            return

        if isinstance(history, dict):
            for metric, values in history.items():
                if isinstance(values, list) and len(values) > 0:
                    print(f"\n{metric}:")
                    print(f"  Initial: {values[0]:.6f}")
                    print(f"  Final:   {values[-1]:.6f}")
                    print(f"  Best:    {min(values):.6f}" if "loss" in metric else f"  Best:    {max(values):.6f}")
                    
                    if len(values) >= 2:
                        change = values[0] - values[-1]
                        pct = abs(change) / abs(values[0]) * 100 if values[0] != 0 else 0
                        direction = "decreased" if change > 0 else "increased"
                        print(f"  Change:  {direction} by {pct:.1f}%")

        print("\n" + "=" * 50)
