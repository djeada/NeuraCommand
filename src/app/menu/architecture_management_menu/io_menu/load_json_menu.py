import json
from pathlib import Path

from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.network_builder.network_builder import NetworkBuilder


class LoadJsonMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu):
        steps = [
            self.prompt_for_path,
            self.load_json,
        ]
        super().__init__(
            parent_menu=parent_menu, name="Load Network from JSON File", steps=steps
        )
        self.file_path = None

    def prompt_for_path(self):
        """Prompt the user to enter the file path and store it."""
        file_path = input("Enter the path to the JSON file: ").strip()

        # Validate the file path
        path = Path(file_path)
        if not path.exists():
            print("File not found. Please enter a valid path.")
            return None  # Skip next steps if path is invalid
        if path.suffix.lower() != ".json":
            print("Please provide a path to a '.json' file.")
            return None  # Skip next steps if file is not JSON

        self.file_path = path

    def load_json(self):
        """Load the JSON file from the stored path."""
        if self.file_path:  # Ensure the file path was set
            try:
                with open(self.file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    self.result = NetworkBuilder.from_json(json_config=data)
                    print(f"Successfully loaded network from {self.file_path}")
            except json.JSONDecodeError as e:
                print(f"Error loading JSON file: {e}")
                self.logger.error(f"JSON decode error for file {self.file_path}: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")
                self.logger.error(f"Error loading file {self.file_path}: {e}")

            self.deactivate()  # Deactivate after attempting to load the file
