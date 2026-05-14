from app.menu.architecture_management_menu.io_menu.load_json_menu import LoadJsonMenu
from app.menu.architecture_management_menu.modify_architecture_menu.modify_architecture_menu import (
    ModifyArchitectureMenu,
)
from app.menu.menu_interface import AbstractMenu, ChoiceBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class ArchitectureManagementMenu(ChoiceBasedMenu):
    def __init__(self, parent_menu: AbstractMenu):
        super().__init__(
            parent_menu=parent_menu,
            name="Architecture Management Menu",
            menu_options={
                "1": {
                    "text": "Create a new neural network architecture",
                    "action": self.create_new_architecture,
                },
                "2": {
                    "text": "Load an existing neural network architecture",
                    "action": self.load_existing_architecture,
                },
                "back": {
                    "text": "Return to the parent menu",
                    "action": self.deactivate,
                },
            },
        )
        self.builder = None

    def create_new_architecture(self):
        self.builder = NetworkBuilder()
        sub_menu = ModifyArchitectureMenu(
            parent_menu=self.parent_menu, network_builder=self.builder
        )
        sub_menu.run()
        self.deactivate()

    def load_existing_architecture(self):
        load_network_menu = LoadJsonMenu(parent_menu=self.parent_menu)
        load_network_menu.run()
        network_architecture = load_network_menu.result
        if network_architecture:
            sub_menu = ModifyArchitectureMenu(
                parent_menu=self.parent_menu, network_builder=network_architecture
            )
            sub_menu.run()
        self.deactivate()
