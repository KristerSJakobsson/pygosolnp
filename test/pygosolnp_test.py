import unittest
from src.pygosolnp import solve
from src.benchmarks.electron import Electron

class TestStringMethods(unittest.TestCase):

    def test_electron(self):
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
                      random_number_seed=443)

        print(result)
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
        self.assertEqual('foo'.upper(), 'FOO')


if __name__ == '__main__':
    unittest.main()