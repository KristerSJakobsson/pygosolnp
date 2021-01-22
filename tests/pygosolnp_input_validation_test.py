import itertools
import unittest

from pygosolnp.pygosolnp import solve
from pygosolnp.sampling import NormalDistribution
from .resources import alkyla_equality_function, parameter_lower_bounds, parameter_upper_bounds, \
    alkyla_inequality_function, alkyla_objective_function, inequality_lower_bounds, inequality_upper_bounds, \
    equality_values


class TestPygosolnpInputValidation(unittest.TestCase):

    def test_bad_objective_function(self):
        # Missing mandatory data
        with self.assertRaises(ValueError):
            solve(obj_func=None,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds)

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=None,
                  par_upper_limit=parameter_upper_bounds)

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=None)

        # Non-callable objective function
        with self.assertRaises(ValueError):
            solve(obj_func={"hello": "world"},
                  par_lower_limit=[1, 2, 3],
                  par_upper_limit=parameter_upper_bounds)

        # Lower bounds missing length attribute
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=2,
                  par_upper_limit=parameter_upper_bounds)

        # Upper bounds missing length attribute
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=1515,
                  par_upper_limit=parameter_upper_bounds)

        # Different length of bounds
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=[1, 2, 3],
                  par_upper_limit=parameter_upper_bounds)

        # Random number distribution not same length as parameters
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  start_guess_sampling=[NormalDistribution(1, 0.1)])

    def test_bad_eq_constraints(self):
        # Eq function without values
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  start_guess_sampling=[NormalDistribution(1, 0.1)],
                  eq_func=alkyla_equality_function)

        # Eq values without function
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  start_guess_sampling=[NormalDistribution(1, 0.1)],
                  eq_values=equality_values)

        # Eq func not callable without function
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  start_guess_sampling=[NormalDistribution(1, 0.1)],
                  eq_func={"Not", "Callable"},
                  eq_values=equality_values)

        # Eq values not of fixed length
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  start_guess_sampling=[NormalDistribution(1, 0.1)],
                  eq_func=alkyla_equality_function,
                  eq_values=2)

    def test_bad_ineq_constraints(self):
        # Ineq func without bounds
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function)

        # Ineq lower bound without func or upper bound
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_lower_bounds=inequality_lower_bounds)

        # Ineq func not callable
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=[1, 2, 3],
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds)

        # Ineq lower bounds not of fixed size
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=itertools.cycle("abcd"),
                  ineq_upper_bounds=inequality_upper_bounds)

        # Ineq upper bounds not of fixed size
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=itertools.cycle("efgh"))

        # Ineq lower and upper bounds of different length
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=[1, 2, 3],
                  ineq_upper_bounds=inequality_upper_bounds)

    def test_bad_pygosolnp_parameters(self):
        # Needs strictly more than 0 simulations
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  number_of_simulations=0)

        # Needs strictly more than 0 restarts
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  number_of_restarts=0)

        # Needs strictly more than 0 processes
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  number_of_processes=0)

    def test_bad_pysolnp_parameters(self):
        # Various pysolnp parameters as wrong variable type
        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_tolerance="a")

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_max_minor_iter="a")

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_max_major_iter="a")

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_rho="a")

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_delta="a")

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  debug="a")

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_max_minor_iter=0)

        with self.assertRaises(ValueError):
            solve(obj_func=alkyla_objective_function,
                  par_lower_limit=parameter_lower_bounds,
                  par_upper_limit=parameter_upper_bounds,
                  ineq_func=alkyla_inequality_function,
                  ineq_lower_bounds=inequality_lower_bounds,
                  ineq_upper_bounds=inequality_upper_bounds,
                  eq_func=alkyla_equality_function,
                  eq_values=equality_values,
                  pysolnp_max_major_iter=0)

