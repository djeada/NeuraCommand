from app.menu.architecture_management_menu.modify_architecture_menu.layer_level_menus.add_layer_menu import (
    AddLayerMenu,
)
from app.menu.architecture_management_menu.modify_architecture_menu.layer_level_menus.select_layer_menu import (
    SelectLayerMenu,
)
from app.menu.architecture_management_menu.modify_architecture_menu.layer_level_menus.remove_layer_menu import (
    RemoveLayerMenu,
)
from app.menu.architecture_management_menu.io_menu.save_json_menu import SaveNetworkMenu
from app.menu.menu_interface import ChoiceBasedMenu, AbstractMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class ModifyArchitectureMenu(ChoiceBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        super().__init__(
            parent_menu=parent_menu,
            name="Modify Architecture Menu",
            menu_options={
                "1": {
                    "text": "Add a new layer to the neural network",
                    "action": self.add_layer,
                },
                "2": {
                    "text": "Remove an existing layer from the network",
                    "action": self.remove_layer,
                },
                "3": {
                    "text": "Change the configuration of a specific layer",
                    "action": self.modify_layer,
                },
                "4": {
                    "text": "Display the current structure of the neural network",
                    "action": self.show_network,
                },
                "5": {
                    "text": "Save the current network configuration to a file",
                    "action": self.save_network,
                },
                "back": {
                    "text": "Return to the parent menu",
                    "action": self.deactivate,
                },
            },
        )
        self.builder = network_builder

    def add_layer(self):
        add_layer_menu = AddLayerMenu(parent_menu=self, network_builder=self.builder)
        add_layer_menu.run()

    def remove_layer(self):
        remove_layer_menu = RemoveLayerMenu(
            parent_menu=self, network_builder=self.builder
        )
        remove_layer_menu.run()

    def modify_layer(self):
        modify_layer_menu = SelectLayerMenu(
            parent_menu=self, network_builder=self.builder
        )
        modify_layer_menu.run()

    def save_network(self):
        save_network_menu = SaveNetworkMenu(
            parent_menu=self, network_builder=self.builder
        )
        save_network_menu.run()

    def show_network(self):
        print(self.builder.to_json())
