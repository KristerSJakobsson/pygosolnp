import unittest
from math import sqrt
from pygosolnp.evaluation_functions import penalty_barrier_function
from pygosolnp.pygosolnp import solve, EvaluationType
from pygosolnp.sampling import NormalDistribution, UniformDistribution
from numpy.random import Generator, MT19937

from .resources import alkyla_equality_function, parameter_lower_bounds, parameter_upper_bounds, \
    alkyla_inequality_function, alkyla_objective_function, inequality_lower_bounds, inequality_upper_bounds, \
    equality_values


class TestPygosolnpFeatures(unittest.TestCase):

    def test_rng(self):
        # Run this twice and assure results are different
        result1 = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds)

        result2 = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds)

        self.assertNotEqual(str(result1), str(result2))

        # RNG with seed
        seed = 15

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        seed_or_generator=seed)

        self.assertAlmostEqual(results.best_solution.obj_value, -172.64079000732818, 4)

        # RNG with generator
        seed = 19937
        generator = Generator(MT19937(seed))

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        seed_or_generator=generator)

        self.assertAlmostEqual(results.best_solution.obj_value, -109.53861769639923, 4)

    def test_evaluation_type(self):
        seed = 123

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        seed_or_generator=seed,
                        evaluation_type=EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ)

        self.assertAlmostEqual(results.best_solution.obj_value, -172.63323139949568, 4)

        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        seed_or_generator=seed,
                        evaluation_type=EvaluationType.PENALTY_BARRIER_FUNCTION.value)

        self.assertAlmostEqual(results.best_solution.obj_value, -137.9176421990594, 4)

    def test_number_of_restarts(self):
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        number_of_restarts=2)

        self.assertEqual(len(results.all_results), 2)

    def test_number_of_simulations(self):
        number_of_simulations = 5
        results = solve(obj_func=alkyla_objective_function,
                        par_lower_limit=parameter_lower_bounds,
                        par_upper_limit=parameter_upper_bounds,
                        eq_func=alkyla_equality_function,
                        eq_values=equality_values,
                        ineq_func=alkyla_inequality_function,
                        ineq_lower_bounds=inequality_lower_bounds,
                        ineq_upper_bounds=inequality_upper_bounds,
                        number_of_simulations=number_of_simulations)

        # We have one guess for each parameter in each simulation
        expected_number_of_starting_guesses = number_of_simulations * len(parameter_lower_bounds)

        self.assertEqual(len(results.starting_guesses), expected_number_of_starting_guesses)

    def test_distribution_settings(self):
        seed = 123
        random_number_distribution = [
            UniformDistribution(lower=parameter_lower_bounds[0], upper=parameter_upper_bounds[0]),
            UniformDistribution(lower=parameter_lower_bounds[1], upper=parameter_upper_bounds[1]),
            UniformDistribution(lower=parameter_lower_bounds[2], upper=parameter_upper_bounds[2]),
            UniformDistribution(lower=parameter_lower_bounds[3], upper=parameter_upper_bounds[3]),
            UniformDistribution(lower=parameter_lower_bounds[4], upper=parameter_upper_bounds[4]),
            UniformDistribution(lower=parameter_lower_bounds[5], upper=parameter_upper_bounds[5]),
            UniformDistribution(lower=parameter_lower_bounds[6], upper=parameter_upper_bounds[6]),
            NormalDistribution(mean=parameter_lower_bounds[7] + (parameter_upper_bounds[7] - parameter_lower_bounds[7]) / 2,
                               standard_deviation=sqrt((parameter_upper_bounds[7] - parameter_lower_bounds[7]) / 4)),
            NormalDistribution(mean=parameter_lower_bounds[8] + (parameter_upper_bounds[8] - parameter_lower_bounds[8]) / 2,
                               standard_deviation=sqrt((parameter_upper_bounds[8] - parameter_lower_bounds[8]) / 4)),
            NormalDistribution(mean=parameter_lower_bounds[9] + (parameter_upper_bounds[9] - parameter_lower_bounds[9]) / 2,
                               standard_deviation=sqrt((parameter_upper_bounds[9] - parameter_lower_bounds[9]) / 4))
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
                        seed_or_generator=seed,
                        random_number_distribution=random_number_distribution)

        expected = [13.647037264962869, 0.8611363008355628, 26.443184732713366, 17.374872427946787, 3.518118021700607, 91.49675605324619, 88.48432483229979, 8.31292833061544, 2.2258202965678078, 152.83537781076353, 10.259409104590638, 3.9194336171007436, 98.90899153168935, 18.550518535003818, 14.829341044694193, 90.03952163671745, 88.8296169746392, 6.510045879312269, 2.2299777335277287, 154.19632889293732, 4.63111249634135, 2.654463891851913, 59.73467621973526, 33.308985624612795, 3.6867597485695947, 85.11915933408186, 50.046324455689174, 9.111045932957914, 2.8400196254592185, 153.51054342599818, 18.342451450952048, 13.835044017500397, 26.17714478999855, 44.64509722952972, 14.615038727425077, 87.22292232239143, 77.74870202059448, 10.142491854767716, 3.3602436241573503, 152.89901318403273, 1.429736132467938, 9.331814562416929, 28.54876795555411, 40.59854584414594, 3.472632723508051, 87.50193805000613, 11.230330527155692, 6.910822616145085, 4.48312018675311, 152.01943163448047, 2.5538064578495545, 4.121000078552381, 0.3817331151973935, 25.242709912632762, 11.51746168073933, 88.4183901702636, 80.9836995212284, 5.267738080122615, 2.6873272124847385, 155.52934383463958, 9.989735006060345, 12.140965136061885, 67.93069035920627, 27.497614462290365, 7.9230888883484685, 85.17788230335903, 49.89481699427682, 9.547662048055379, 2.880185054883278, 152.03280311481785, 9.712828061599943, 8.305842303671929, 49.03091759905391, 33.15182875404989, 1.4070134655145061, 88.9070706496258, 61.86231040564725, 10.055864173423558, 2.2844820374118444, 152.46974188625146, 0.11780066634939468, 12.103663522335316, 9.30911642382917, 29.59952150635655, 6.087221935649126, 91.72657729741088, 90.79044829125715, 6.528052183506991, 2.2545933363624022, 151.44848414573949, 16.242242210271527, 12.781497568945824, 78.63422124132948, 19.148138191024216, 2.753489294845368, 88.39496911141609, 22.880794006641082, 9.473498559840532, 1.4414176562849212, 152.87413041845966]
        self.assertListEqual(results.starting_guesses, expected)
