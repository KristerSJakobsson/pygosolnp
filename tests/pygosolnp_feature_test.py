import unittest

from pygosolnp.pygosolnp import solve, EvaluationType
from pygosolnp.sampling import NormalDistribution, UniformDistribution, TriangleDistribution, ConstantValue
from tests.resources import alkyla_equality_function, parameter_lower_bounds, parameter_upper_bounds, \
    alkyla_inequality_function, alkyla_objective_function, inequality_lower_bounds, inequality_upper_bounds, \
    equality_values


class TestPygosolnpFeatures(unittest.TestCase):

    def test_rng(self):
        # Run this twice and assure results are different
        result1 = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds)

        result2 = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds)

        self.assertNotEqual(str(result1), str(result2))

        # RNG with seed, note that we remove the constraints to the problem to make it easier to find a solution
        seed = 15

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        seed=seed)

        self.assertAlmostEqual(results.best_solution.obj_value, -2597.805528666477, 4)

    def test_evaluation_type(self):
        seed = 19937

        # Eval Type OBJECTIVE_FUNC_EXCLUDE_INEQ has a hard time locating starting points for problems with narrow inequality bounds
        # Alkyla is a bad function in this sense, but with enough simulations we find something...
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
                        number_of_simulations=1000000)

        self.assertAlmostEqual(results.best_solution.obj_value, -172.6396637275493, 4)

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

        self.assertAlmostEqual(results.best_solution.obj_value, -172.6344504852143, 4)

    def test_number_of_restarts(self):
        number_of_restarts = 2
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        number_of_restarts=number_of_restarts)

        self.assertEqual(len(results.all_results), 2)

    def test_number_of_simulations(self):
        number_of_simulations = 5
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        number_of_simulations=number_of_simulations)

        # We have one guess for each parameter in each simulation
        expected_number_of_starting_guesses = number_of_simulations * len(parameter_lower_bounds)

        self.assertEqual(len(results.starting_guesses), expected_number_of_starting_guesses)

    def test_distribution_settings(self):
        # Test the built-in distributions:
        ## UniformDistribution (default)
        ## TriangleDistribution
        ## ConstantValue (not really a distribution, but anyway...)
        ## NormalDistribution (not bound to the problem limits, so can cause ValueErrors)
        # For examples with truncated normal distribution or grid sampling, see the pygosolnp/python_examples/ folder.

        seed = 123
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

        expected = [1.0472719770188865, 1.394986840362117, 48.8690116404478, 14.308009397537562, 18.02397755903389, 85.30522932881858, 55.57717340288379, 6.667970391053073, 3.6, 156.49983212864856, 6.744333142185511, 5.340742314063284, 29.419602302113333, 10.066822143168913, 8.725515868304367, 85.70090799800339, 60.79595448147919, 4.682265418742503, 3.6, 149.48291056761096, 6.309179474717266, 7.1731473329111, 108.64783827413987, 13.710582455139804, 2.8435012886300215, 91.3205747548704, 11.805057859444085, 10.10305434913332, 3.6, 147.52272080665128, 16.75675288390269, 12.300715959679737, 41.21545897871575, 42.05986365168132, 4.137249413322229, 89.87122020952442, 54.80137093372041, 9.22118914786812, 3.6, 150.52935280055388, 6.305263044319762, 6.371810423486387, 92.31349078967085, 30.5100853545626, 13.69674437322012, 90.10682611444399, 66.6658986412204, 9.890258808939956, 3.6, 140.91723345075388, 7.4904326364496665, 0.5094609078576298, 90.50078062306466, 17.258596636601634, 16.403803688676618, 92.30916729491383, 65.93630337909174, 5.624907417542909, 3.6, 154.59882922969987, 6.738323055834123, 6.259092913140844, 8.794893374842388, 28.270682808058147, 11.463370290210628, 87.68884879447867, 26.0702589141365, 4.889542390237933, 3.6, 152.554473932165, 7.038273661964023, 3.6660762316845474, 63.839698881811756, 38.946534044769656, 18.18306551273559, 90.11040671394481, 94.30241347891283, 10.793516049964698, 3.6, 152.51421058231693, 12.599814194484914, 3.7339709453785535, 58.0573779171776, 17.103589222956202, 19.839155455884338, 88.55599936119259, 73.89026343234005, 4.336884126714421, 3.6, 157.19018629693437, 5.12724694593085, 5.632760487194819, 32.63158227921019, 27.37074334450615, 8.407146334701082, 85.86896667585282, 15.321745860406931, 6.792095542121678, 3.6, 157.67128088975167]
        self.assertListEqual(results.starting_guesses, expected)
