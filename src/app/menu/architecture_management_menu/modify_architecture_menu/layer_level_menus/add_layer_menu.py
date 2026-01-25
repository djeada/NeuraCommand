from app.menu.architecture_management_menu.modify_architecture_menu.perceptron_level_menus.add_perceptron_menu import (
    AddPerceptronMenu,
)
from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class AddLayerMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        steps = [
            self.get_num_perceptrons,
            self.get_same_config_answer,
            self.create_layer_config,
            self.add_layer_to_network,
        ]
        super().__init__(
            parent_menu=parent_menu, name="Add New Layer to Network", steps=steps
        )
        self.network_builder = network_builder
        self.num_perceptrons = None
        self.same_config = None
        self.layer_config = None

    def get_num_perceptrons(self):
        self.num_perceptrons = int(
            input("Enter the number of perceptrons in this layer: ")
        )
        return self.num_perceptrons

    def get_same_config_answer(self):
        answer = (
            input(
                "Use the same configuration for all perceptrons? (yes/no) [default: yes]: "
            )
            .lower()
            .strip()
        )

        if answer == "":
            answer = (
                "yes"  # Set default value to "yes" if user presses Enter without input
            )

        while answer not in ["yes", "y", "no", "n"]:
            answer = (
                input(
                    "Use the same configuration for all perceptrons? (yes/no) [default: yes]: "
                )
                .lower()
                .strip()
            )
            if answer == "":
                answer = "yes"  # Set default value to "yes" if user presses Enter without input

        self.same_config = answer
        return self.same_config

    def create_layer_config(self):
        if self.same_config in ["yes", "y"]:
            perceptron_config = self.get_perceptron_config()
            self.layer_config = {
                "perceptrons": [perceptron_config] * self.num_perceptrons
            }
        else:
            self.layer_config = self.get_individual_perceptron_configs(
                self.num_perceptrons
            )
        return self.layer_config

    def get_perceptron_config(self):
        add_perceptron_menu = AddPerceptronMenu(parent_menu=self)
        add_perceptron_menu.run()
        return add_perceptron_menu.result

    def get_individual_perceptron_configs(self, num_perceptrons: int):
        layer_config = {"perceptrons": []}
        for i in range(num_perceptrons):
            print(f"\nConfiguring Perceptron {i + 1}")
            perceptron_config = self.get_perceptron_config()
            layer_config["perceptrons"].append(perceptron_config)
        return layer_config

    def add_layer_to_network(self):
        try:
            self.network_builder.add_layer(self.layer_config)
            print(f"Layer added successfully. {self.network_builder.get_layer(-1)}")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
