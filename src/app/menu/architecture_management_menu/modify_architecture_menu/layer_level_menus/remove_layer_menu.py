from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class RemoveLayerMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        steps = [self.display_current_layers, self.confirm_removal]
        super().__init__(
            parent_menu=parent_menu, name="Remove Layer from Network", steps=steps
        )
        self.network_builder = network_builder
        self.layer_to_remove = None

    def display_current_layers(self):
        """Displays the current layers and gets the index of the layer to remove."""
        print("\nRemoving a layer from the network")
        num_layers = len(self.network_builder.config["layers"])

        if num_layers == 0:
            print("No layers to remove.")
            self.deactivate()
        else:
            print(f"Current number of layers: {num_layers}")
            self.layer_to_remove = int(
                input("Enter the index of the layer to remove (0-based index): ")
            )

    def confirm_removal(self):
        """Removes the specified layer if the index is valid."""
        num_layers = len(self.network_builder.config["layers"])
        if self.layer_to_remove is not None and 0 <= self.layer_to_remove < num_layers:
            try:
                self.network_builder.remove_layer(self.layer_to_remove)
                print(f"Layer at index {self.layer_to_remove} removed successfully.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        else:
            print(
                f"Invalid index. Please enter a number between 0 and {num_layers - 1}."
            )
        self.deactivate()
