import argparse
import math
from abc import ABC, abstractmethod
from pathlib import Path


class BaseCommand(ABC):
    def __init__(self, args):
        self.args = args

    @staticmethod
    @abstractmethod
    def add_arguments(parser):
        pass

    @abstractmethod
    def execute(self):
        """Subclasses must implement this method."""
        pass

    @staticmethod
    def file_path(path):
        if not Path(path).is_file():
            raise argparse.ArgumentTypeError(
                f"The path {path} must be an existing file."
            )
        return path

    @staticmethod
    def positive_int(value):
        number = BaseCommand.nonnegative_int(value)
        if number == 0:
            raise argparse.ArgumentTypeError("must be a positive integer")
        return number

    @staticmethod
    def nonnegative_int(value):
        try:
            number = int(value)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if number < 0:
            raise argparse.ArgumentTypeError("must be a non-negative integer")
        return number

    @staticmethod
    def seed(value):
        number = BaseCommand.nonnegative_int(value)
        if number >= 2**32:
            raise argparse.ArgumentTypeError("seed must be smaller than 2**32")
        return number

    @staticmethod
    def nonnegative_float(value):
        try:
            number = float(value)
        except (ValueError, TypeError) as error:
            raise argparse.ArgumentTypeError("must be a number") from error
        if not math.isfinite(number) or number < 0:
            raise argparse.ArgumentTypeError("must be finite and non-negative")
        return number

    @staticmethod
    def positive_float(value):
        number = BaseCommand.nonnegative_float(value)
        if number == 0:
            raise argparse.ArgumentTypeError("must be positive")
        return number

    @staticmethod
    def validation_fraction(value):
        number = BaseCommand.nonnegative_float(value)
        if number >= 1:
            raise argparse.ArgumentTypeError("must be in the range [0, 1)")
        return number

    @staticmethod
    def probability(value):
        number = BaseCommand.nonnegative_float(value)
        if number > 1:
            raise argparse.ArgumentTypeError("must be in the range [0, 1]")
        return number

    @staticmethod
    def precision(value):
        number = BaseCommand.nonnegative_int(value)
        if number > 16:
            raise argparse.ArgumentTypeError("precision must be between 0 and 16")
        return number
