import json
import os
import re
from random import Random


class MockRandom(Random):
    """
    This class uses the logic of random and only replacs the uniform normal generation logic with a pre-generated file
    The purpose is to guarantee the same random numbers accross different os, architecture and Python versions
    """
    def __init__(self, x=None):
        mock_random_number_data = MockRandom.load_random_number_data()

        if x is None:
            raise Exception(f"Mock random number generation requires seed!")

        mock_random_numbers = {data["seed"]: data["samples"] for data in mock_random_number_data}

        if x not in mock_random_numbers:
            raise Exception(f"Missing random mock data for seed {x} of type {type(x)}")

        self.mock_random_number_sequence = mock_random_numbers[x]
        self.random_number_index = 0
        super().__init__(x)

    @staticmethod
    def load_random_number_data():
        """ RNG data is chunked due to GitHub limiting files to 100 MB"""
        folder_path = os.path.dirname(__file__)
        files = os.listdir(folder_path)
        mock_random_number_data_raw = ""
        regex = r"^random_numbers(\d).json$"  # File names for chunked RNG data

        sorted_files = []
        for file in files:
            match = re.match(pattern=regex, string=file)
            if match:
                index = int(match[1])
                sorted_files.append((file, index))

        sorted_files = sorted(sorted_files, key=lambda x: x[1])

        for file, index in sorted_files:
            match = re.match(pattern=regex, string=file)
            if match:
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r") as input_data:
                    mock_random_number_data_raw = mock_random_number_data_raw + input_data.read()

        mock_random_number_data = json.loads(mock_random_number_data_raw)
        return mock_random_number_data

    def random(self):
        if self.random_number_index > len(self.mock_random_number_sequence) - 1:
            raise Exception(f"Missing random number for index {self.random_number_index}, sequence of length {len(self.mock_random_number_sequence)}")

        result = self.mock_random_number_sequence[self.random_number_index]
        self.random_number_index = self.random_number_index + 1

        return result
