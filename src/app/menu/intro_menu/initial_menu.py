"""Initial menu for the NeuraCommand interactive CLI."""

import sys

from app.menu.architecture_management_menu.architecture_management_menu import (
    ArchitectureManagementMenu,
)
from app.menu.menu_interface import ChoiceBasedMenu
from app.menu.model_management_menu.model_management_menu import ModelManagementMenu


class InitialMenu(ChoiceBasedMenu):
    """Main entry point menu for the interactive CLI application."""

    def __init__(self):
        super().__init__(
            name="Neura Command CLI Application",
            menu_options={
                "1": {
                    "text": "Architecture Management",
                    "action": self.architecture_management,
                },
                "2": {"text": "Model Management", "action": self.model_management},
                "3": {"text": "Exit", "action": self.exit_app},
            },
        )

    def architecture_management(self):
        """Open the architecture management submenu."""
        sub_menu = ArchitectureManagementMenu(parent_menu=self)
        sub_menu.run()

    def model_management(self):
        """Open the model management submenu."""
        sub_menu = ModelManagementMenu(parent_menu=self)
        sub_menu.run()

    def exit_app(self):
        """Exit the application."""
        sys.exit(0)
