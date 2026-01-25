from app.menu.menu_interface import AbstractMenu, StepBasedMenu


class AddPerceptronMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu):
        steps = [
            self.set_input_size,
            self.set_output_size,
            self.set_activation_function,
            self.set_loss_function,
            self.set_learning_rate,
            self.complete_configuration,
        ]
        super().__init__(
            parent_menu=parent_menu, name="Configure New Perceptron", steps=steps
        )
        self.perceptron_config = {}

    def set_input_size(self):
        self.perceptron_config["input_size"] = int(
            input("Enter input size for Perceptron: ")
        )

    def set_output_size(self):
        self.perceptron_config["output_size"] = int(
            input("Enter output size for Perceptron: ")
        )

    def set_activation_function(self):
        self.perceptron_config["activation_func"] = (
            self.get_activation_function_choice()
        )

    def set_loss_function(self):
        self.perceptron_config["loss_func"] = self.get_loss_function_choice()

    def set_learning_rate(self):
        self.perceptron_config["learning_rate"] = float(
            input("Enter learning rate for Perceptron (default 0.1): ") or 0.1
        )

    def complete_configuration(self):
        try:
            # Additional validation can be done here
            print("Perceptron configured successfully.")
            self.result = self.perceptron_config
            self.deactivate()
        except Exception as e:
            print(f"Error in configuration: {e}")
            self.deactivate()

    def get_activation_function_choice(self):
        options = {
            "1": "sigmoid",
            "2": "basicsigmoid",
            "3": "relu",
            "4": "step",
            "5": "tanh",
            "6": "linear",
        }
        print("Select an activation function:")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice: ")
        return options.get(choice, "linear").lower()

    def get_loss_function_choice(self):
        options = {"1": "difference", "2": "mse", "3": "crossentropy", "4": "huber"}
        print("Select a loss function:")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice: ")
        return options.get(choice, "mse").lower()
