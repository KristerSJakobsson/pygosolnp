import unittest

from src.evaluation_functions import penalty_barrier_function


class TestUtilityFunctions(unittest.TestCase):

    def test_penalty_barrier_function(self):
        # Use the alkyla function from pysolnp benchmarks to test lagrangian function
        def alkyla_objective_function(x):
            result = -0.63 * x[3] * x[6] + 50.4 * x[0] + 3.5 * x[1] + x[2] + 33.6 * x[4]
            return result

        def alkyla_equality_function(x):
            result = [
                98.0 * x[2] - 0.1 * x[3] * x[5] * x[8] - x[2] * x[5],
                1000.0 * x[1] + 100.0 * x[4] - 100.0 * x[0] * x[7],
                122.0 * x[3] - 100.0 * x[0] - 100.0 * x[4],
            ]
            return result

        def alkyla_inequality_function(x):
            result = [
                (1.12 * x[0] + 0.13167 * x[0] * x[7] - 0.00667 * x[0] * x[7] * x[7]) / x[3],
                (1.098 * x[7] - 0.038 * x[7] * x[7] + 0.325 * x[5] + 57.25) / x[6],
                (-0.222 * x[9] + 35.82) / x[8],
                (3.0 * x[6] - 133.0) / x[9],
            ]
            return result

        equality_values = [0, 0, 0]
        inequality_lower_bounds = [0.99, 0.99, 0.9, 0.99]
        inequality_upper_bounds = [100 / 99, 100 / 99, 10 / 9, 100 / 99]

        testing_point = [17.45, 12.0, 110.0, 30.0, 19.74, 89.2, 92.8, 8.0, 3.6, 155.0]

        value = penalty_barrier_function(testing_point, obj_func=alkyla_objective_function,
                                         eq_func=alkyla_equality_function, eq_values=equality_values,
                                         ineq_func=alkyla_inequality_function,
                                         ineq_lower_bounds=inequality_lower_bounds,
                                         ineq_upper_bounds=inequality_upper_bounds)
        self.assertAlmostEqual(value, 348.8093, 4)


if __name__ == '__main__':
    unittest.main()
