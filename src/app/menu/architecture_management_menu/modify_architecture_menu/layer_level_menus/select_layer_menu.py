from app.menu.architecture_management_menu.modify_architecture_menu.layer_level_menus.modify_layer_menu import (
    LayerMenu,
)
from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class SelectLayerMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        steps = [self.display_num_layers, self.prompt_layer_index, self.modify_layer]
        super().__init__(
            parent_menu=parent_menu, name="Modify Layer in Network", steps=steps
        )
        self.network_builder = network_builder
        self.layer_index = None

    def display_num_layers(self):
        """Displays the number of layers and sets up for layer index selection."""
        print("\n--- Modifying a Layer in the Network ---")
        num_layers = len(self.network_builder.config["layers"])

        if num_layers == 0:
            print("No layers to modify.")
            self.deactivate()

    def prompt_layer_index(self):
        """Prompts the user to enter the index of the layer to modify."""
        num_layers = len(self.network_builder.config["layers"])
        if num_layers > 0:
            print(
                f"Enter the index of the layer to modify (0-based index). Currently, there are {num_layers} layers."
            )
            try:
                self.layer_index = int(input("Layer index: "))
                if not (0 <= self.layer_index < num_layers):
                    print(
                        f"Invalid layer index. Please enter a number between 0 and {num_layers - 1}."
                    )
                    self.deactivate()  # Invalid choice, deactivate and exit
            except ValueError:
                print("Invalid input. Please enter a valid number.")
                self.deactivate()  # Invalid input, deactivate and exit

    def modify_layer(self):
        """Initiates the modification of the selected layer."""
        if self.layer_index is not None:
            layer_menu = LayerMenu(
                parent_menu=self,
                network_builder=self.network_builder,
                layer_index=self.layer_index,
            )
            layer_menu.run()
        self.deactivate()  # Deactivate after modifying the layer
