import pickle


class FeedForwardNetworkIO:
    @staticmethod
    def save_to_pickle(network, file_path):
        """Saves a FeedForwardNetwork object to a file."""
        with open(file_path, "wb") as file:
            pickle.dump(network, file)

    @staticmethod
    def load_from_pickle(file_path):
        """Loads a FeedForwardNetwork object from a file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)
