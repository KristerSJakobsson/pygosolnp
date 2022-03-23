############################
# This example shows how to run pygosolnp with Truncated Normal distribution using Numpy and Scipy
############################

from typing import List, Optional
import time

# Numpy random has the PCG64 generator which according to some research is better than Mersenne Twister
from numpy.random import Generator, PCG64

# Note that this script depends on scipy, which is not a requirement for pygosolnp
from scipy.stats import truncnorm

import pygosolnp


# The Sampling class is an abstract class that can be inherited and customized as you please
class TruncatedNormalSampling(pygosolnp.sampling.Sampling):

    def __init__(self,
                 parameter_lower_bounds: List[float],
                 parameter_upper_bounds: List[float],
                 seed: Optional[int]):
        self.__generator = Generator(PCG64(seed))
        self.__parameter_lower_bounds = parameter_lower_bounds
        self.__parameter_upper_bounds = parameter_upper_bounds

    def generate_sample(self, sample_size: int) -> List[float]:
        # This function returns random starting values for one sample
        return truncnorm.rvs(a=self.__parameter_lower_bounds,
                             b=self.__parameter_upper_bounds,
                             size=sample_size,
                             random_state=self.__generator)


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
    sampling = TruncatedNormalSampling(
        parameter_lower_bounds=parameter_lower_bounds,
        parameter_upper_bounds=parameter_upper_bounds,
        seed=99)

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

# Solution 1: 0.0016119745327847497; Solution 2: 0.005968645850086645; Solution 3: 0.006083292803668321; Solution 4: 0.006629107105976147; Solution 5: 0.005305936314073526; Solution 6: 0.006049589559946693
# Best solution: [1.3008954298086124, 3.181786909056148, 1.3814249752478918, 3.9436695447632877]
# Objective function value: 0.0016119745327847497
# Elapsed time: 8.562503099441528 s

# Quicker than Grid Sampling but not as accurate in the same number of optimization attempts.
