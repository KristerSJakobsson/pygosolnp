from random import Random

# This script generates random numbers based on seeds and stores them to random_numbers.json
# The reason for this is to provide the same RNG output for tests regardless of operating system or Python version

import json
import os
from random import Random


def generate():
    sample_data = [
        {
            "seed": 15,
            "number_of_samples": 200000 * 10
        },
        {
            "seed": 123,
            "number_of_samples": 10 * 10
        },
        {
            "seed": 442,
            "number_of_samples": 20000 * 3 * 25
        },
        {
            "seed": 443,
            "number_of_samples": 20000 * 3 * 25
        },
        {
            "seed": 1234567,
            "number_of_samples": 10000000
        }
    ]

    for sample in sample_data:
        seed = sample["seed"]
        number_of_samples = sample["number_of_samples"]
        generator = Random(seed)
        sample_numbers = []
        for index in range(number_of_samples):
            sample_numbers.append(generator.random())
        sample["samples"] = sample_numbers

    data_string = json.dumps(sample_data)
    number_of_chunks = 6
    data_length = len(data_string)
    chunk_size = data_length // number_of_chunks
    chunks = list(range(0, data_length, chunk_size))

    start_index = 0
    final_index = data_length

    start_indices = chunks
    end_indices = chunks[1:] + [final_index]
    intervals = zip(start_indices, end_indices)

    for index, interval in enumerate(intervals):
        path = os.path.join(os.path.dirname(__file__), f"random_numbers{index}.json")
        with open(path, "w") as output_data:
            output_data.write(data_string[interval[0]: interval[1]])


if __name__ == "__main__":
    generate()
