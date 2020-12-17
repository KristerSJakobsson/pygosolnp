import abc
import random
from typing import Optional, List, Union


class Distribution(abc.ABC):

    @abc.abstractmethod
    def generate(self, generator: random.Random) -> float:
        pass


class UniformDistribution(Distribution):

    def __init__(self, lower: float, upper: float):
        self.__lower = lower
        self.__upper = upper

    def generate(self, generator: random.Random) -> float:
        return generator.uniform(self.__lower, self.__upper)


class NormalDistribution(Distribution):

    def __init__(self, mean, standard_deviation):
        self.__mean = mean
        self.__standard_deviation = standard_deviation

    def generate(self, generator: random.Random) -> float:
        return generator.gauss(self.__mean, self.__standard_deviation)


class Sampling(abc.ABC):

    def generate_all_samples(self, number_of_samples: int, sample_size: int) -> List[float]:
        samples: List[Union[float, None]] = [None] * number_of_samples * sample_size
        for sample_index in range(number_of_samples):
            samples[sample_index * sample_size: (sample_index + 1) * sample_size] = self.generate_sample(
                sample_size=sample_size)

        return samples

    @abc.abstractmethod
    def generate_sample(self, sample_size: int):
        pass


class DefaultSampling(Sampling):

    def __init__(self,
                 parameter_lower_bounds: List[float],
                 parameter_upper_bounds: List[float],
                 sample_properties: Optional[List[Distribution]],
                 seed: int):

        self.__generator = random.Random(seed)

        if type(sample_properties) is list:
            self.__sample_properties = sample_properties
        else:
            default_sample_properties = [UniformDistribution(lower=lower, upper=upper) for lower, upper in
                                         zip(parameter_lower_bounds, parameter_upper_bounds)]
            self.__sample_properties = default_sample_properties

    def generate_sample(self, sample_size: int) -> List[float]:
        result = [0.0] * sample_size
        for variable_index, distribution in enumerate(self.__sample_properties):
            result[variable_index] = distribution.generate(generator=self.__generator)
        return result
