import unittest
from unittest.mock import patch

from pygosolnp.benchmarks.electron import Electron, obj_func, eq_func
from pygosolnp.pygosolnp import solve, EvaluationType

from tests.mock.mock_random import MockRandom


class TestPygosolnpElectron(unittest.TestCase):
    electron = Electron(number_of_charges=25)

    def test_electron_functions(self):
        optimum = [0.61433646, 0.51219787, -0.76783671, -0.99899687, -0.17556954, -0.33268115,
                   -0.03210551, -0.76888194, -0.74949967, 0.81150788, -0.72320933, -0.68025229,
                   0.45952132, 0.93840065, 0.68628241, 0.32797284, -0.10528078, 0.13167533,
                   0.92064585, 0.47118278, -0.16734767, -0.37218115, -0.31940340, -0.04146434,
                   0.36161351, -0.02432836, 0.64347663, 0.58093602, -0.02911027, 0.81443792,
                   0.92618849, 0.24959548, -0.40666050, 0.41394741, -0.26006846, -0.61183818,
                   0.04054085, -0.67193437, 0.33057648, 0.50458088, 0.06385934, -0.94738933,
                   -0.60190580, -0.35666799, -0.86096396, -0.45402367, -0.04542384, -0.88241957,
                   0.65598514, 0.92970571, -0.78866904, -0.56885074, -0.27007430, -0.03402711,
                   -0.55305172, 0.17747712, -0.96781782, 0.49340430, 0.51662151, 0.52327751,
                   -0.32034716, -0.73185604, -0.58081353, -0.10061517, 0.52384596, 0.94252629,
                   -0.30227374, 0.78763635, -0.15874245, 0.19164511, -0.87513271, 0.92704793,
                   0.34542317, 0.75363400, 0.06987819]

        objective_function = lambda x: self.electron.objective_function(x)
        equality_function = lambda x: self.electron.equality_function(x)
        equality_bounds = self.electron.equality_constraint_bounds
        upper_bounds = self.electron.parameter_upper_bound
        lower_bounds = self.electron.parameter_lower_bound

        objective_function_value = objective_function(optimum)
        equality_function_value = equality_function(optimum)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        self.assertAlmostEqual(objective_function_value, 243.8128, 3)

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 6)

    @patch(target="random.Random", new=MockRandom)
    def test_electron_optimization_exclude_inequalities_single_process(self):
        objective_function = lambda x: self.electron.objective_function(x)
        equality_function = lambda x: self.electron.equality_function(x)
        equality_bounds = self.electron.equality_constraint_bounds
        upper_bounds = self.electron.parameter_upper_bound
        lower_bounds = self.electron.parameter_lower_bound

        results = solve(obj_func=objective_function,
                        eq_func=equality_function,
                        eq_values=equality_bounds,
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
        equality_function_value = equality_function(optimum)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 3)

        self.assertLessEqual(results.best_solution.obj_value, 244)

    @patch(target="random.Random", new=MockRandom)
    def test_electron_optimization_penalty_barrier_function_single_process(self):
        objective_function = lambda x: self.electron.objective_function(x)
        equality_function = lambda x: self.electron.equality_function(x)
        equality_bounds = self.electron.equality_constraint_bounds
        upper_bounds = self.electron.parameter_upper_bound
        lower_bounds = self.electron.parameter_lower_bound

        results = solve(obj_func=objective_function,
                        eq_func=equality_function,
                        eq_values=equality_bounds,
                        par_lower_limit=lower_bounds,
                        par_upper_limit=upper_bounds,
                        number_of_restarts=2,
                        number_of_simulations=20000,
                        number_of_processes=None,
                        seed=443,
                        pysolnp_max_major_iter=100,
                        evaluation_type=EvaluationType.PENALTY_BARRIER_FUNCTION,
                        debug=False)

        optimum = results.best_solution.parameters
        equality_function_value = equality_function(optimum)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 3)

        self.assertLessEqual(results.best_solution.obj_value, 244)

    @patch(target="random.Random", new=MockRandom)
    def test_electron_optimization_exclude_inequalities_multiple_processes(self):
        equality_bounds = self.electron.equality_constraint_bounds
        upper_bounds = self.electron.parameter_upper_bound
        lower_bounds = self.electron.parameter_lower_bound

        results = solve(obj_func=obj_func,
                        eq_func=eq_func,
                        eq_values=equality_bounds,
                        par_lower_limit=lower_bounds,
                        par_upper_limit=upper_bounds,
                        number_of_restarts=20,
                        number_of_simulations=20000,
                        number_of_processes=4,
                        seed=443,
                        pysolnp_max_major_iter=100,
                        evaluation_type=EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
                        debug=False)

        optimum = results.best_solution.parameters
        equality_function_value = eq_func(optimum)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 3)

        self.assertLessEqual(results.best_solution.obj_value, 244)

    @patch(target="random.Random", new=MockRandom)
    def test_electron_optimization_penalty_barrier_function_multiple_processes(self):
        equality_bounds = self.electron.equality_constraint_bounds
        upper_bounds = self.electron.parameter_upper_bound
        lower_bounds = self.electron.parameter_lower_bound

        results = solve(obj_func=obj_func,
                        eq_func=eq_func,
                        eq_values=equality_bounds,
                        par_lower_limit=lower_bounds,
                        par_upper_limit=upper_bounds,
                        number_of_restarts=2,
                        number_of_simulations=20000,
                        number_of_processes=None,
                        seed=443,
                        pysolnp_max_major_iter=100,
                        evaluation_type=EvaluationType.PENALTY_BARRIER_FUNCTION,
                        debug=False)

        optimum = results.best_solution.parameters
        equality_function_value = eq_func(optimum)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 3)

        self.assertLessEqual(results.best_solution.obj_value, 244)
