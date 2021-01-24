############################
# This example shows how to make your own custom distributions for use with pygosolnp
############################

from itertools import chain
from typing import List

# Scikit-optimize has a wide range of useful sampling functions, see below link
# https://scikit-optimize.github.io/stable/auto_examples/sampler/initial-sampling-method.html
import skopt

import pygosolnp


# The Sampling class is an abstract class that can be inherited and customized as you please
class GridSampling(pygosolnp.sampling.Sampling):

    def __init__(self,
                 parameter_lower_bounds: List[float],
                 parameter_upper_bounds: List[float],
                 seed):
        self.__space = skopt.space.Space(dimensions=zip(parameter_lower_bounds, parameter_upper_bounds))
        self.__seed = seed

    def generate_all_samples(self, number_of_samples: int, sample_size: int) -> List[float]:
        # Overwrite this function to define the behavior when generating starting guesses for all samples
        # By default it calls `generate_sample` number_of_samples times, however we customize it here
        grid = skopt.sampler.Grid()
        grid_values = grid.generate(dimensions=self.__space.dimensions,
                                    n_samples=number_of_samples,
                                    random_state=self.__seed)
        return list(chain.from_iterable(grid_values))

    def generate_sample(self, sample_size: int) -> List[float]:
        # This function is abstract in the base class
        # Not needed since we are generating a grid for all samples, so overwrite it with pass
        pass


# The Permutation Function has unique solution f(x) = 0 when x_i = i
def permutation_function(data):
    n = 4
    b = 0.5

    result1 = 0
    for index1 in range(1, n + 1):
        result2 = 0
        for index2 in range(1, n + 1):
            result2 += ((pow(index2, index1) + b) * (pow(data[index2 - 1] / index2, index1) - 1))
        result1 += pow(result2, 2)
    return result1


parameter_lower_bounds = [-4.0] * 4
parameter_upper_bounds = [4.0] * 4

if __name__ == '__main__':
    # Instantiate sampling object
    sampling = GridSampling(
        parameter_lower_bounds=parameter_lower_bounds,
        parameter_upper_bounds=parameter_upper_bounds,
        seed=92)

    # Note that the seed variable to pygosolnp.solve is ignored due to the custom sampling
    results = pygosolnp.solve(
        obj_func=permutation_function,
        par_lower_limit=parameter_lower_bounds,
        par_upper_limit=parameter_upper_bounds,
        number_of_restarts=6,
        number_of_simulations=2000,
        pysolnp_max_major_iter=25,
        pysolnp_tolerance=1E-9,
        start_guess_sampling=sampling)

    print(results.best_solution)

# Best solution: [0.6222222222222218, 2.2222222222222223, 3.822222222222222, 3.2888888888888888]
# Objective function value: 9.91928145483169
# Not perfect, but much better than truncated normal!
