import unittest

from pygosolnp.evaluation_functions import penalty_barrier_function
from tests.resources import alkyla_equality_function, alkyla_inequality_function, alkyla_objective_function, \
    inequality_lower_bounds, inequality_upper_bounds, equality_values


class TestPygosolnpEvaluationFunctions(unittest.TestCase):

    def test_penalty_barrier_function(self):
        testing_point = [17.45, 12.0, 110.0, 30.0, 19.74, 89.2, 92.8, 8.0, 3.6, 155.0]

        value = penalty_barrier_function(testing_point, obj_func=alkyla_objective_function,
                                         eq_func=alkyla_equality_function, eq_values=equality_values,
                                         ineq_func=alkyla_inequality_function,
                                         ineq_lower_bounds=inequality_lower_bounds,
                                         ineq_upper_bounds=inequality_upper_bounds)
        self.assertAlmostEqual(value, 348.8093, 4)
