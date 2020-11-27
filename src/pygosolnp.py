import random
import pysolnp

import multiprocessing as multi

from typing import Callable, List, Optional
from enum import Enum

from src.benchmarks.electron import Electron

def generate_random_value(lower_bound, upper_bound):
    return random.uniform(lower_bound, upper_bound)


class Result:
    def __init__(self):
        pass


class Distribution(Enum):
    UNIFORM = "uniform"


def solve(obj_func: Callable,
          par_start_value: List[float],
          par_lower_limit: List[float] = None,
          par_upper_limit: List[float] = None,
          eq_func: Optional[Callable] = None,
          eq_values: Optional[List[float]] = None,
          ineq_func: Optional[Callable] = None,
          ineq_lower_bounds: Optional[List[float]] = None,
          ineq_upper_bounds: Optional[List[float]] = None,
          fixed_starting_parameters: Optional[List[float]] = None,
          rho: float = 1.0,
          max_major_iter: int = 10,
          max_minor_iter: int = 10,
          delta: float = 1e-05,
          tolerance: float = 0.0001,
          debug: bool = False,
          number_of_processes: Optional[int] = 5,
          random_number_distribution: Distribution = Distribution.UNIFORM,
          random_number_seed: Optional[int] = None) -> Result:

    if type(par_lower_limit) is not List or type(par_upper_limit) is not List:
        raise ValueError("Paramer lower and upper limits are required for gosolnp")

    if len(par_lower_limit) != len(par_upper_limit):
        raise ValueError("The length of lower and upper limits are not the same.")

    # Expand fixed starting parameters if any
    actual_obj_func = obj_func
    actual_eq_func = eq_func
    actual_ineq_func = ineq_func
    if fixed_starting_parameters is not None:
        def modify_parameteres(x: List[float]) -> List[float]:
            original_index = 0
            new_x = fixed_starting_parameters.copy()
            for index in range(len(new_x)):
                if new_x[index] is None:
                    new_x[index] = x[original_index]
                    original_index = original_index + 1
            return new_x

        actual_obj_func = lambda x : obj_func(modify_parameteres(x))

        if eq_func is not None:
            actual_eq_func = lambda x: obj_func(modify_parameteres(x))

        if ineq_func is not None:
            actual_ineq_func = lambda x: obj_func(modify_parameteres(x))

    if random_number_seed is None:
        rng = random.Random()
    else:
        rng = random.Random(random_number_seed)
    number = rng.uniform(0, 1)
    random.seed = random_number_seed
    if number_of_processes is None:
        # Run single-processed
        pass
    else:
        pool = multi.Pool(processes=number_of_processes)



if __name__ == "__main__":


    electron = Electron(number_of_charges=25)

    objective_function = lambda x: electron.objective_function(x)
    equality_function = lambda x: electron.equality_function(x)
    equality_bounds = electron.equality_constraint_bounds
    upper_bounds = electron.parameter_upper_bound
    lower_bounds = electron.parameter_lower_bound

    solve(fixed_starting_parameters=None,
          obj_func=objective_function,
          eq_func=equality_function,
          eq_values=equality_bounds,
          par_lower_limit=lower_bounds,
          par_upper_limit=upper_bounds)

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