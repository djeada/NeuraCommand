import numpy as np

from app.menu.menu_interface import StepBasedMenu, AbstractMenu
from neura_command.deprecated.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)


class MakePredictionMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network: FeedForwardNetwork):
        steps = [
            self.choose_input_method,
            self.load_or_enter_data,
            self.make_prediction,
            self.display_prediction,
            self.save_prediction,
        ]
        super().__init__(parent_menu=parent_menu, name="Make Prediction", steps=steps)
        self.model = network
        self.data = None
        self.prediction = None

    def choose_input_method(self):
        print("\nChoose the method to input data:")
        print("1. Load from CSV")
        print("2. Enter data manually")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            self.method = "csv"
        elif choice == "2":
            self.method = "manual"
        else:
            print("Invalid choice. Please enter 1 or 2.")
            self.deactivate()

    def load_or_enter_data(self):
        if self.method == "csv":
            file_path = input("Enter the path of the CSV file: ").strip()
            self.data = self.load_data_from_csv(file_path)
        elif self.method == "manual":
            data_input = input("Enter your data (comma-separated): ").strip()
            values = np.array([float(x) for x in data_input.split(",")], dtype=float)
            self.data = values.reshape(1, -1)

    def make_prediction(self):
        if self.data is not None:
            self.prediction = self.model.predict(self.data)
            print("Prediction made successfully.")

    def display_prediction(self):
        print("Prediction result:", self.prediction)

    def save_prediction(self):
        save_choice = (
            input("Do you want to save the prediction to a CSV file? (y/n): ")
            .strip()
            .lower()
        )
        if save_choice == "y":
            save_path = input("Enter the path to save the CSV: ").strip()
            self.save_prediction_to_csv(save_path)

    def load_data_from_csv(self, file_path):
        try:
            data = np.genfromtxt(file_path, delimiter=",")
            if data.ndim == 1:
                if data.size == self.model.input_size:
                    data = data.reshape(1, -1)
                elif data.size % self.model.input_size == 0:
                    data = data.reshape(-1, self.model.input_size)
                else:
                    raise ValueError(
                        f"Expected rows with {self.model.input_size} input values, got {data.size} values."
                    )
            elif data.shape[1] != self.model.input_size:
                raise ValueError(
                    f"Expected {self.model.input_size} input columns, got {data.shape[1]}."
                )
            print(f"Data loaded successfully from {file_path}")
            return data
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None

    def save_prediction_to_csv(self, file_path):
        try:
            np.savetxt(file_path, self.prediction, delimiter=",")
            print(f"Prediction saved successfully to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving the prediction: {e}")
