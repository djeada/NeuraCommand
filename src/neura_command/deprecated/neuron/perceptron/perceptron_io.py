import pickle


class PerceptronIO:
    @staticmethod
    def save(perceptron, file_path):
        """Saves a Perceptron object to a file."""
        with open(file_path, "wb") as file:
            pickle.dump(perceptron, file)

    @staticmethod
    def load(file_path):
        """Loads a Perceptron object from a file."""
        with open(file_path, "rb") as file:
            return pickle.load(file)
