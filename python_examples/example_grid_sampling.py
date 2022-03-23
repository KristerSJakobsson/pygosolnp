############################
# This example shows how to make your own custom distributions for use with pygosolnp
############################

import time
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
    start = time.time()

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

    end = time.time()

    all_results = results.all_results
    print("; ".join([f"Solution {index + 1}: {solution.obj_value}" for index, solution in enumerate(all_results)]))
    best_solution = results.best_solution
    print(f"Best solution: {results.best_solution.parameters}")
    print(f"Objective function value: {results.best_solution.obj_value}")
    print(f"Elapsed time: {end - start} s")

# Solution 1: 0.0006360327708392506; Solution 2: 0.006239163594915304; Solution 3: 0.006140229082904356; Solution 4: 0.006218870214655177; Solution 5: 0.005963823643719209; Solution 6: 0.13065649880545976
# Best solution: [1.1622677695732497, 1.683172007310748, 3.9509962074974956, 3.159134907203731]
# Objective function value: 0.0006360327708392506
# Elapsed time: 22.986207962036133 s

# Slower than Truncated Normal but higher precision in the same number of optimization attempts.