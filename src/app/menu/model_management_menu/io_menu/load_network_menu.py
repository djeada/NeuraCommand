"""Load network menu for loading models from disk."""

from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from app.menu.architecture_management_menu.io_menu.load_json_menu import LoadJsonMenu
from app.menu.model_management_menu.io_menu.load_pickle_menu import LoadPickleMenu


class LoadNetworkMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu):
        steps = [self.choose_file_type, self.load_file]
        super().__init__(
            parent_menu=parent_menu, name="Load Network from Disk", steps=steps
        )
        self.file_type = None

    def choose_file_type(self):
        """Prompt the user to choose the file type to load."""
        print("\n--- Load Network from Disk ---")
        print("1. Load from JSON file")
        print("2. Load from Pickle file")
        choice = input("Choose the file type (1 for JSON, 2 for Pickle): ").strip()

        if choice == "1":
            self.file_type = "json"
        elif choice == "2":
            self.file_type = "pickle"
        else:
            print("Invalid choice. Please choose 1 or 2.")
            self.deactivate()

    def load_file(self):
        """Load the file based on the chosen file type."""
        if self.file_type == "json":
            load_json_menu = LoadJsonMenu(parent_menu=self)
            load_json_menu.run()
        elif self.file_type == "pickle":
            load_pickle_menu = LoadPickleMenu(parent_menu=self)
            load_pickle_menu.run()
        self.deactivate()
