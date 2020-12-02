import unittest
from src.pygosolnp import solve
from src.benchmarks.electron import Electron


class TestStringMethods(unittest.TestCase):

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

        number_of_charges = len(optimum) // 3  # Each charge has three coordinates, x: 0-24 y: 25-49 z: 50-74

        electron = Electron(number_of_charges=number_of_charges)

        objective_function = lambda x: electron.objective_function(x)
        equality_function = lambda x: electron.equality_function(x)
        equality_bounds = electron.equality_constraint_bounds
        upper_bounds = electron.parameter_upper_bound
        lower_bounds = electron.parameter_lower_bound

        objective_function_value = objective_function(optimum)
        equality_function_value = equality_function(optimum)
        print(objective_function_value)
        print(equality_function_value)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        self.assertAlmostEqual(objective_function_value, 243.8128, 4)

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 6)

    def test_electron_optimization(self):
        electron = Electron(number_of_charges=25)

        objective_function = lambda x: electron.objective_function(x)
        equality_function = lambda x: electron.equality_function(x)
        equality_bounds = electron.equality_constraint_bounds
        upper_bounds = electron.parameter_upper_bound
        lower_bounds = electron.parameter_lower_bound

        result = solve(fixed_starting_parameters=None,
                       obj_func=objective_function,
                       eq_func=equality_function,
                       eq_values=equality_bounds,
                       par_lower_limit=lower_bounds,
                       par_upper_limit=upper_bounds,
                       number_of_restarts=2,
                       number_of_simulations=20000,
                       number_of_processes=None,
                       random_number_seed=443,
                       debug=True)

        optimum = result.optimum
        objective_function_value = objective_function(optimum)
        equality_function_value = equality_function(optimum)
        print(objective_function_value)
        print(equality_function_value)

        for index, value in enumerate(optimum):
            self.assertGreaterEqual(value, lower_bounds[index])
            self.assertLessEqual(value, upper_bounds[index])

        for index, value in enumerate(equality_function_value):
            self.assertAlmostEqual(value, equality_bounds[index], 6)

        #
        # self.assertAlmostEqual(result.solve_value, 243.813, 3)
        # # should get a function value around 243.813

        # ans = gosolnp(pars=NULL, fixed=NULL, fun=gofn, eqfun=goeqfn, eqB=eqB, LB=LB, UB=UB,
        #               control=list(), distr=rep(1, length(LB)), distr.opt = list(outer.iter = 10, trace = 1),
        # n.restarts = 2, n.sim = 20000, rseed = 443, n = 25)
        #

        # bt = data.frame(solnp=rbind(round(ans$values[length(ans$values)], 5L),
        # round(ans$outer.iter, 0L),
        # round(ans$convergence, 0L),
        # round(ans$nfuneval, 0L),
        # round(ans$elapsed, 3L),
        # matrix(round(ans$pars, 5L), ncol = 1L)),
        # conopt = rbind(round(conopt$fn, 5L),
        # round(conopt$iter, 0L),
        # round(0, 0L),
        # round(conopt$nfun, 0L),
        # round(conopt$elapsed, 3L),
        # matrix(round(conopt$pars, 5L), ncol = 1L)) )
        # rownames(bt) < - c("funcValue", "majorIter", "exitFlag", "nfunEval", "time(sec)",
        #                    paste("par.", 1L: length(ans$pars), sep = "") )
        # colnames(bt) = c("solnp", "conopt")
        # attr(bt, "description") = paste(
        #     "The equilibrium state distribution (of minimal Coulomb potential)\n of the electrons positioned on a conducting sphere.")
        # return (bt)


if __name__ == '__main__':
    unittest.main()
