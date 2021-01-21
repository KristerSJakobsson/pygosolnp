import unittest
from unittest.mock import patch

from pygosolnp.pygosolnp import solve, EvaluationType
from pygosolnp.sampling import NormalDistribution, UniformDistribution, TriangleDistribution, ConstantValue
from tests.resources import alkyla_equality_function, parameter_lower_bounds, parameter_upper_bounds, \
    alkyla_inequality_function, alkyla_objective_function, inequality_lower_bounds, inequality_upper_bounds, \
    equality_values
from tests.mock.mock_random import MockRandom


class TestPygosolnpFeatures(unittest.TestCase):

    def test_rng_without_seed(self):
        # Note: No need to patch random since we are expecting actual random outcome in this test
        # Run this twice and assure results are different
        result1 = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds)

        result2 = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds)

        self.assertNotEqual(str(result1), str(result2))

    @patch(target="random.Random", new=MockRandom)
    def test_rng_with_seed(self):
        # RNG with seed, note that we remove the constraints to the problem to make it easier to find a solution
        seed = 15

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        seed=seed)

        self.assertAlmostEqual(results.best_solution.obj_value, -2597.805528666477, 4)

    @patch(target="random.Random", new=MockRandom)
    def test_evaluation_type(self):
        seed = 1234567

        # Eval Type OBJECTIVE_FUNC_EXCLUDE_INEQ has a hard time locating starting points for problems with narrow inequality bounds
        # Alkyla is a bad function in this sense, but with enough luck and enough simulations we find something...
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        seed=seed,
                        evaluation_type=EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
                        number_of_simulations=100000)

        self.assertAlmostEqual(results.best_solution.obj_value, -172.64110132537394, 4)

        # Eval Type PENALTY_BARRIER_FUNCTION is better at locating starting points for problems with narrow inequality bounds
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        seed=seed,
                        evaluation_type=EvaluationType.PENALTY_BARRIER_FUNCTION.value)

        self.assertAlmostEqual(results.best_solution.obj_value, -170.98167226891587, 4)

    @patch(target="random.Random", new=MockRandom)
    def test_number_of_restarts(self):
        number_of_restarts = 2
        seed = 1234567
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        number_of_restarts=number_of_restarts,
                        seed=seed)

        self.assertEqual(len(results.all_results), 2)

    @patch(target="random.Random", new=MockRandom)
    def test_number_of_simulations(self):
        number_of_simulations = 5
        seed = 1234567
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        number_of_simulations=number_of_simulations,
                        seed=seed)

        # We have one guess for each parameter in each simulation
        expected_number_of_starting_guesses = number_of_simulations * len(parameter_lower_bounds)

        self.assertEqual(len(results.starting_guesses), expected_number_of_starting_guesses)

    @patch(target="random.Random", new=MockRandom)
    def test_distribution_settings(self):
        # Test the built-in distributions:
        ## UniformDistribution (default)
        ## TriangleDistribution
        ## ConstantValue (not really a distribution, but anyway...)
        ## NormalDistribution (not bound to the problem limits, so can cause ValueErrors)
        # For examples with truncated normal distribution or grid sampling, see the pygosolnp/python_examples/ folder.

        seed = 1234567
        random_number_distribution = [
            UniformDistribution(lower=parameter_lower_bounds[0], upper=parameter_upper_bounds[0]),
            UniformDistribution(lower=parameter_lower_bounds[1], upper=parameter_upper_bounds[1]),
            UniformDistribution(lower=parameter_lower_bounds[2], upper=parameter_upper_bounds[2]),
            UniformDistribution(lower=parameter_lower_bounds[3], upper=parameter_upper_bounds[3]),
            UniformDistribution(lower=parameter_lower_bounds[4], upper=parameter_upper_bounds[4]),
            UniformDistribution(lower=parameter_lower_bounds[5], upper=parameter_upper_bounds[5]),
            UniformDistribution(lower=parameter_lower_bounds[6], upper=parameter_upper_bounds[6]),
            TriangleDistribution(lower=parameter_lower_bounds[7],
                                 upper=parameter_upper_bounds[7],
                                 mode=(parameter_upper_bounds[7] + parameter_lower_bounds[7]) / 2),
            ConstantValue(value=3.6),  # Forced to always be 3.6
            NormalDistribution(mean=parameter_lower_bounds[9] + (parameter_upper_bounds[9] - parameter_lower_bounds[9]) / 2,
                               standard_deviation=(parameter_upper_bounds[9] - parameter_lower_bounds[9]) / 2)
        ]

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        number_of_simulations=10,
                        number_of_restarts=1,
                        seed=seed,
                        evaluation_type=EvaluationType.PENALTY_BARRIER_FUNCTION,
                        start_guess_sampling=random_number_distribution)

        expected = [19.262864952192448, 3.771726139520572, 69.77797447275604, 19.068793973427482, 19.33182084216137, 89.89207643509036, 16.299077190999093, 7.656803716342179, 3.6, 141.68084366746555, 16.527810282976148, 2.8024116596314954, 113.8847499219252, 41.913648375763266, 13.653863317763928, 89.36107961371863, 31.65003670354735, 6.742247466242993, 3.6, 144.41878023451304, 9.155674825062048, 10.587247668990255, 89.6903787978092, 38.29483391669842, 8.064799536849511, 89.43255250817325, 86.41114587298019, 5.961162104738718, 3.6, 147.6264662441887, 10.067591751749159, 13.136604255021913, 73.43265943827133, 44.724776804281476, 0.7740903344904604, 88.462319551243, 50.21684783727512, 5.77672041074722, 3.6, 145.39269004124904, 9.068225746042762, 12.61405383313014, 68.82767844281483, 16.611623195615945, 9.00169037647699, 85.31447609370339, 51.318693144764346, 9.28229782560638, 3.6, 150.88899802298727, 2.853206753025248, 13.650346506177705, 12.332917949430415, 43.013069327734556, 19.840074115550717, 91.20142486328595, 79.33118000010684, 3.8956025310633953, 3.6, 153.4160006648588, 0.6434958413807812, 2.9463400863302613, 102.90608010105203, 38.77132597397568, 3.023003386685923, 85.51893369450241, 59.20378581486704, 4.856217794173671, 3.6, 159.0155564965776, 19.650096860859275, 14.557888535778906, 24.793341705879453, 28.499566032377153, 12.098785988152493, 92.47510246654802, 44.847907830159485, 7.819085343574262, 3.6, 154.12786686990134, 8.266201078057069, 10.758463796133531, 0.2800364042800263, 23.509199466708623, 6.6730557487734306, 86.50613863644213, 65.27792963956803, 10.344369491252513, 3.6, 153.0090154093769, 13.044272915070119, 3.896854881633315, 86.85241703973514, 22.201841915524042, 17.718325429729628, 89.780805965076, 64.98014550674284, 8.237553597176323, 3.6, 150.8605530994913]
        self.assertListEqual(results.starting_guesses, expected)
