from app.menu.architecture_management_menu.modify_architecture_menu.perceptron_level_menus.modify_perceptron_menu import (
    ModifyPerceptronMenu,
)
from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class SelectPerceptronMenu(StepBasedMenu):
    def __init__(
        self,
        parent_menu: AbstractMenu,
        network_builder: NetworkBuilder,
        layer_index: int,
    ):
        steps = [
            self.display_num_perceptrons,
            self.prompt_perceptron_index,
            self.modify_perceptron,
        ]
        super().__init__(
            parent_menu=parent_menu,
            name=f"Modifying a Perceptron in Layer {layer_index}",
            steps=steps,
        )
        self.network_builder = network_builder
        self.layer_index = layer_index
        self.perceptron_index = None

    def display_num_perceptrons(self):
        """Displays the number of perceptrons and sets up for perceptron index selection."""
        print(f"\n--- Modifying a Perceptron in Layer {self.layer_index} ---")
        num_perceptrons = self.get_num_perceptrons()

        if num_perceptrons == 0:
            print("No perceptrons to modify in this layer.")
            self.deactivate()

    def prompt_perceptron_index(self):
        """Prompts the user to enter the index of the perceptron to modify."""
        num_perceptrons = self.get_num_perceptrons()
        if num_perceptrons > 0:
            print(
                f"Enter the index of the perceptron to modify (0-based index). Currently, there are {num_perceptrons} perceptrons."
            )
            try:
                self.perceptron_index = int(input("Perceptron index: "))
                if not (0 <= self.perceptron_index < num_perceptrons):
                    print(
                        f"Invalid perceptron index. Please enter a number between 0 and {num_perceptrons - 1}."
                    )
                    self.deactivate()  # Invalid choice, deactivate and exit
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                self.deactivate()  # Invalid input, deactivate and exit

    def modify_perceptron(self):
        """Initiates the modification of the selected perceptron."""
        if self.perceptron_index is not None:
            modify_perceptron_menu = ModifyPerceptronMenu(
                parent_menu=self,
                network_builder=self.network_builder,
                layer_index=self.layer_index,
                perceptron_index=self.perceptron_index,
            )
            modify_perceptron_menu.run()
        self.deactivate()  # Deactivate after modifying the perceptron

    def get_num_perceptrons(self) -> int:
        """Returns the number of perceptrons in the selected layer."""
        layer = self.network_builder.get_layer(self.layer_index)
        return len(layer["perceptrons"])
