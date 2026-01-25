from pathlib import Path
import json

from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class SaveNetworkMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu, network_builder: NetworkBuilder):
        steps = [
            self.prompt_for_path,
            self.confirm_overwrite,
            self.save_network_configuration,
        ]
        super().__init__(parent_menu=parent_menu, name="Save Network Menu", steps=steps)
        self.network_builder = network_builder
        self.file_path = None

    def prompt_for_path(self):
        default_path = "network.json"
        file_path = input(
            f"Enter the file path to save the network configuration (default: {default_path}): "
        ).strip()

        self.file_path = Path(file_path) if file_path else Path(default_path)

    def confirm_overwrite(self):
        if self.file_path.exists():
            overwrite = input("File already exists. Overwrite? (y/n): ").lower()
            if overwrite != "y" and overwrite != "":
                print("Save operation cancelled.")
                self.deactivate()  # Stop further steps
        else:
            print("No conflicts with existing files detected.")

    def save_network_configuration(self):
        try:
            with self.file_path.open("w") as file:
                # Directly save the dictionary as JSON
                json.dump(self.network_builder.to_json(), file, indent=4)
            print(f"Network configuration saved successfully to {self.file_path}")
            self.deactivate()  # Deactivate after saving
        except Exception as e:
            print(f"An error occurred while saving the file: {e}")
            self.deactivate()
