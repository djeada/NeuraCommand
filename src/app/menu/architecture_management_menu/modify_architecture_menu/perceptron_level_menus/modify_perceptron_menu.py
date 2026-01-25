from app.menu.menu_interface import AbstractMenu, ChoiceBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class ModifyPerceptronMenu(ChoiceBasedMenu):
    def __init__(
        self,
        parent_menu: AbstractMenu,
        network_builder: NetworkBuilder,
        layer_index: int,
        perceptron_index: int,
    ):
        menu_options = {
            "1": {"text": "Modify input size", "action": self.modify_input_size},
            "2": {"text": "Modify output size", "action": self.modify_output_size},
            "3": {
                "text": "Change activation function",
                "action": self.change_activation_function,
            },
            "4": {"text": "Change loss function", "action": self.change_loss_function},
            "5": {"text": "Adjust learning rate", "action": self.adjust_learning_rate},
            "back": {"text": "Return to the previous menu", "action": self.deactivate},
        }
        super().__init__(
            parent_menu=parent_menu,
            name="Modify Perceptron Menu",
            menu_options=menu_options,
        )
        self.network_builder = network_builder
        self.layer_index = layer_index
        self.perceptron_index = perceptron_index

    def modify_input_size(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_input_size = int(
            input(f"Enter new input size (current: {perceptron['input_size']}): ")
        )
        perceptron["input_size"] = new_input_size
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Input size updated.")

    def modify_output_size(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_output_size = int(
            input(f"Enter new output size (current: {perceptron['output_size']}): ")
        )
        perceptron["output_size"] = new_output_size
        self.network_builder.set_perceptron(
            self.layer_index, self.perceptron_index, perceptron
        )
        print("Output size updated.")

    def change_activation_function(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_activation_func = self.get_activation_function_choice(
            current=perceptron["activation_func"]
        )
        perceptron["activation_func"] = new_activation_func
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Activation function updated.")

    def change_loss_function(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_loss_func = self.get_loss_function_choice(current=perceptron["loss_func"])
        perceptron["loss_func"] = new_loss_func
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Loss function updated.")

    def adjust_learning_rate(self):
        perceptron = self.network_builder.get_perceptron(
            layer_index=self.layer_index, perceptron_index=self.perceptron_index
        )
        new_learning_rate = float(
            input(f"Enter new learning rate (current: {perceptron['learning_rate']}): ")
        )
        perceptron["learning_rate"] = new_learning_rate
        self.network_builder.set_perceptron(
            layer_index=self.layer_index,
            perceptron_index=self.perceptron_index,
            new_perceptron_config=perceptron,
        )
        print("Learning rate updated.")

    def get_activation_function_choice(self, current: str):
        options = {
            "1": "sigmoid",
            "2": "basicsigmoid",
            "3": "relu",
            "4": "step",
            "5": "tanh",
            "6": "linear",
        }

        print(f"Select an activation function (current: {current}):")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice (leave blank for no change): ")
        if choice.strip() == "":
            return current
        return options.get(choice, current).lower()

    def get_loss_function_choice(self, current: str):
        options = {"1": "difference", "2": "mse", "3": "crossentropy", "4": "huber"}

        print(f"Select a loss function (current: {current}):")
        for key, value in options.items():
            print(f"[{key}] {value.capitalize()}")

        choice = input("Enter choice (leave blank for no change): ")
        if choice.strip() == "":
            return current
        return options.get(choice, current).lower()
