from pathlib import Path

from app.menu.menu_interface import AbstractMenu, StepBasedMenu
from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
    FeedForwardNetworkIO,
)


class LoadPickleMenu(StepBasedMenu):
    def __init__(self, parent_menu: AbstractMenu):
        steps = [self.prompt_for_path, self.load_network_from_pickle]
        super().__init__(
            parent_menu=parent_menu, name="Load Network from Pickle File", steps=steps
        )
        self.file_path = None

    def prompt_for_path(self):
        """Prompt the user to enter the path of the pickle file."""
        print("\nLoad Network from Pickle File")
        print("Enter the path to the pickle file (or 'back' to return):")
        path_input = input().strip()

        if path_input.lower() == "back":
            self.deactivate()
            return

        self.file_path = Path(path_input)
        if not self.file_path.exists() or self.file_path.suffix.lower() != ".pickle":
            print(
                "Invalid file path or file type. Please enter a valid .pickle file path."
            )
            self.deactivate()

    def load_network_from_pickle(self):
        """Load the network from the provided pickle file path."""
        if self.file_path:
            try:
                print(f"Loading network from {self.file_path}...")
                network = FeedForwardNetworkIO.load_from_pickle(self.file_path)
                self.result = network  # 'result' is defined in the base class
                print("Network loaded successfully.")
            except Exception as e:
                print(f"Failed to load network: {e}")
        self.deactivate()  # Deactivate after attempting to load the file
