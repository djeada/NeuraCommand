import numpy as np

from app.menu.menu_interface import StepBasedMenu, AbstractMenu
from neura_command.deprecated.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)


class TrainNetworkMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network: FeedForwardNetwork):
        steps = [
            self.choose_data_input_method,
            self.load_or_enter_training_data,
            self.set_training_parameters,
            self.execute_training,
        ]
        super().__init__(parent_menu=parent_menu, name="Train Network", steps=steps)
        self.network = network
        self.data_method = None
        self.X_train, self.y_train = None, None
        self.epochs, self.batch_size = None, None

    def choose_data_input_method(self):
        print("\nChoose how to input training data:")
        print("1. Load from CSV")
        print("2. Enter data manually")
        while True:
            self.data_method = input("Enter your choice (1 or 2): ").strip()
            if self.data_method in ["1", "2"]:
                break
            print("Invalid choice. Please enter 1 or 2.")

    def load_or_enter_training_data(self):
        if self.data_method == "1":
            self.X_train, self.y_train = self.load_data_from_csv()
        elif self.data_method == "2":
            self.X_train, self.y_train = self.enter_data_manually()

    def set_training_parameters(self):
        while True:
            try:
                self.epochs = int(input("Enter the number of epochs: "))
                self.batch_size = int(input("Enter the batch size: "))
                if self.epochs <= 0 or self.batch_size <= 0:
                    print("Epochs and batch size must be positive integers.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter integer values.")

    def execute_training(self):
        print("Training the network...")
        # Ensure data and parameters are set
        if self.X_train is not None and self.y_train is not None:
            self.network.train(
                self.X_train,
                self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
            )
            print("Training completed.")
        else:
            print(
                "Training data or parameters are missing. Please restart the training process."
            )

    def load_data_from_csv(self):
        file_path = input("Enter the path of the training CSV file: ").strip()
        try:
            data = np.genfromtxt(file_path, delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if (
                data.ndim < 2
                or data.shape[1] < self.network.input_size + self.network.output_size
            ):
                print(
                    f"Invalid CSV format. Ensure the CSV has at least {self.network.input_size + self.network.output_size} columns."
                )
                return None, None

            X_train = data[:, : self.network.input_size]
            y_train = data[
                :,
                self.network.input_size : self.network.input_size
                + self.network.output_size,
            ]

            return X_train, y_train
        except Exception as e:
            print(f"An error occurred while loading the data: {e}")
            return None, None

    def enter_data_manually(self):
        print(
            f"Enter your data row by row ({self.network.input_size} features and {self.network.output_size} output values separated by commas), end with 'done':"
        )
        data_entries = []
        while True:
            row = input()
            if row.lower() == "done":
                break
            try:
                row_data = [float(value) for value in row.split(",")]
                if len(row_data) != self.network.input_size + self.network.output_size:
                    print(
                        f"Each row must contain {self.network.input_size + self.network.output_size} values."
                    )
                    continue
                data_entries.append(row_data)
            except ValueError:
                print("Invalid input. Ensure all values are numbers.")

        if not data_entries:
            print("No training data entered.")
            return None, None

        data_array = np.array(data_entries)
        X_train = data_array[:, : self.network.input_size]
        y_train = data_array[:, self.network.input_size :]
        return X_train, y_train
