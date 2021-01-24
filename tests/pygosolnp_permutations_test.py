import unittest
from unittest.mock import patch

from pygosolnp.benchmarks.permutations import permutation_function, parameter_lower_bounds, parameter_upper_bounds
from pygosolnp.pygosolnp import solve, EvaluationType

from tests.mock.mock_random import MockRandom


class TestPygosolnpPermutation(unittest.TestCase):

    def test_permutations_ideal_solution_functions(self):
        # This is the theoretical optimum
        optimum = [1.0, 2.0, 3.0, 4.0]
        objective_function = permutation_function
        objective_function_value = objective_function(optimum)
        self.assertAlmostEqual(objective_function_value, 0.0, 5)

    def test_permutations_close_solution_functions(self):
        # This is the solution we get from RSOLNP
        optimum = [1.000085, 1.999737, 3.000211, 3.999942]
        objective_function = permutation_function
        objective_function_value = objective_function(optimum)
        self.assertAlmostEqual(objective_function_value, 1.883935e-08, 5)

    @patch(target="random.Random", new=MockRandom)
    def test_permuation_optimization_exclude_indequalities_single_process(self):
        objective_function = permutation_function
        upper_bounds = parameter_upper_bounds
        lower_bounds = parameter_lower_bounds

        results = solve(obj_func=objective_function,
                        par_lower_limit=lower_bounds,
                        par_upper_limit=upper_bounds,
                        number_of_restarts=20,
                        number_of_simulations=20000,
                        number_of_processes=None,
                        seed=443,
                        pysolnp_max_major_iter=100,
                        evaluation_type=EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
                        debug=False)

        optimum = results.best_solution.parameters

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        self.assertLessEqual(results.best_solution.obj_value, 0.5)

    @patch(target="random.Random", new=MockRandom)
    def test_permutations_optimization_penalty_barrier_function_single_process(self):
        objective_function = permutation_function
        upper_bounds = parameter_upper_bounds
        lower_bounds = parameter_lower_bounds

        results = solve(obj_func=objective_function,
                        par_lower_limit=lower_bounds,
                        par_upper_limit=upper_bounds,
                        number_of_restarts=20,
                        number_of_simulations=20000,
                        number_of_processes=None,
                        seed=443,
                        pysolnp_max_major_iter=100,
                        evaluation_type=EvaluationType.PENALTY_BARRIER_FUNCTION,
                        debug=False)

        optimum = results.best_solution.parameters

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        self.assertLessEqual(results.best_solution.obj_value, 0.5)
