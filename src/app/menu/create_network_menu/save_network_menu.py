import json
from pathlib import Path

from app.menu.menu_interface import Menu
from app.network_builder.network_builder import NetworkBuilder


class SaveNetworkMenu(Menu):
    def __init__(self, parent_menu: Menu, network_builder: NetworkBuilder):
        super().__init__(parent_menu=parent_menu)
        self.network_builder = network_builder
        self.options = {
            "1": self.view_network_configuration,
            "2": self.save_network_configuration,
            "back": self.go_back,
        }

    def display(self):
        print("\nNetwork Configuration Menu")
        print("1. View Current Network Configuration")
        print("2. Save Network Configuration to Disk")
        print("Enter 'back' to return to the main menu or 'exit' to quit.")

    def view_network_configuration(self):
        print("\nCurrent Network Configuration:")
        print(json.dumps(self.network_builder.to_json(), indent=4))
        self.display()

    def save_network_configuration(self):
        file_path = input("Enter the file path to save the network configuration: ")
        path = Path(file_path)

        if path.exists():
            overwrite = input("File already exists. Overwrite? (y/n): ")
            if overwrite.lower() != "y":
                print("Save operation cancelled.")
                return

        self._write_to_file(path)

    def _write_to_file(self, path: Path):
        try:
            with path.open("w") as file:
                json_config = json.dumps(self.network_builder.to_json(), indent=4)
                file.write(json_config)
            print(f"Network configuration saved successfully to {path}")
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")

    def go_back(self):
        self.parent_menu.run()

    def handle_selection(self, choice):
        if choice.lower() in self.options:
            self.options[choice.lower()]()
        else:
            print("Invalid choice. Please try again.")
            self.display()
