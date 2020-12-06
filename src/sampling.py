import abc
from typing import List, Optional

from numpy.random import Generator


class Distribution(abc.ABC):

    @abc.abstractmethod
    def generate(self, generator: Generator) -> float:
        pass


class UniformDistribution(Distribution):

    def __init__(self, lower: float, upper: float):
        self.__lower = lower
        self.__upper = upper

    def generate(self, generator: Generator) -> float:
        return generator.uniform(self.__lower, self.__upper)


class NormalDistribution(Distribution):

    def __init__(self, mean, standard_deviation):
        self.__mean = mean
        self.__standard_deviation = standard_deviation

    def generate(self, generator: Generator) -> float:
        return generator.normal(self.__mean, self.__standard_deviation)


class Sampling:

    def __init__(self, lower_bounds: List[float],
                 upper_bounds: List[float],
                 sample_properties: Optional[List[Distribution]],
                 generator: Generator):
        self.__generator = generator

        if type(sample_properties) is list:
            self.__sample_properties = sample_properties
        else:
            default_sample_properties = [UniformDistribution(lower=lower, upper=upper) for lower, upper in zip(lower_bounds, upper_bounds)]
            self.__sample_properties = default_sample_properties

    def generate_samples(self, number_of_samples: int) -> List[Optional[float]]:
        samples = [None] * number_of_samples * len(self.__sample_properties)
        for sample_index in range(number_of_samples):
            for variable_index, distribution in enumerate(self.__sample_properties):
                samples[sample_index * variable_index] = distribution.generate(generator=self.__generator)
        return samples
