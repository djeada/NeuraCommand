from pathlib import Path

from app.menu.menu_interface import StepBasedMenu, AbstractMenu
from neura_command.deprecated.feed_forward_network.feed_forward_network import (
    FeedForwardNetwork,
)
from neura_command.deprecated.feed_forward_network.feed_forward_network_io import (
    FeedForwardNetworkIO,
)


class SavePickleMenu(StepBasedMenu):
    DEFAULT_SUFFIX = ".pkl"

    def __init__(self, parent_menu: AbstractMenu, network: FeedForwardNetwork):
        steps = [self.prompt_for_path, self.save_network_to_pickle]
        super().__init__(
            parent_menu=parent_menu, name="Save Network to Pickle File", steps=steps
        )
        self.file_path = None
        self.network = network

    def prompt_for_path(self):
        """Prompt the user to enter the path where the pickle file will be saved."""
        print("\nSave Network to Pickle File")
        print("Enter the path to save the pickle file (or 'back' to return):")
        path_input = input().strip()

        if path_input.lower() == "back":
            self.deactivate()
            return

        self.file_path = Path(path_input)
        if self.file_path.suffix == "":
            self.file_path = self.file_path.with_suffix(self.DEFAULT_SUFFIX)
        if self.file_path.is_dir():
            print("Invalid file path. Please enter a valid file path, not a directory.")
            self.deactivate()

    def save_network_to_pickle(self):
        """Save the network to the provided pickle file path."""
        if self.file_path:
            try:
                print(f"Saving network to {self.file_path}...")
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                FeedForwardNetworkIO.save_to_pickle(self.network, self.file_path)
                print("Network saved successfully to pickle.")
            except Exception as e:
                print(f"Failed to save network: {e}")
        self.deactivate()  # Deactivate after attempting to save the file
