import multiprocessing as multi
from functools import reduce
from heapq import nsmallest
from multiprocessing import Array, Value
from typing import Callable, List, Optional
from ctypes import c_int, c_double, c_bool

import pysolnp
from numpy.random import Generator, MT19937

from src import resources
from src.evaluation_functions import evaluate_starting_guess, pysolnp_solve
from src.model import ProblemModel, EvaluationType
from src.sampling import Distribution, Sampling


def solve(obj_func: Callable,
          par_lower_limit: List[float],
          par_upper_limit: List[float],
          number_of_restarts: int,
          number_of_simulations: int,
          eq_func: Optional[Callable] = None,
          eq_values: Optional[List[float]] = None,
          ineq_func: Optional[Callable] = None,
          ineq_lower_bounds: Optional[List[float]] = None,
          ineq_upper_bounds: Optional[List[float]] = None,
          rho: float = 1.0,
          max_major_iter: int = 10,
          max_minor_iter: int = 10,
          delta: float = 1e-05,
          tolerance: float = 0.0001,
          debug: bool = False,
          number_of_processes: Optional[int] = 5,
          random_number_distribution: Optional[List[Distribution]] = None,
          random_number_seed: Optional[int] = None,
          evaluation_type: EvaluationType = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ) -> pysolnp.Result:
    model = ProblemModel(obj_func=obj_func,
                         par_lower_limit=par_lower_limit,
                         par_upper_limit=par_upper_limit,
                         number_of_restarts=number_of_restarts,
                         number_of_simulations=number_of_simulations,
                         eq_func=eq_func,
                         eq_values=eq_values,
                         ineq_func=ineq_func,
                         ineq_lower_bounds=ineq_lower_bounds,
                         ineq_upper_bounds=ineq_upper_bounds,
                         rho=rho,
                         max_major_iter=max_major_iter,
                         max_minor_iter=max_minor_iter,
                         delta=delta,
                         tolerance=tolerance,
                         debug=debug,
                         number_of_processes=number_of_processes,
                         random_number_distribution=random_number_distribution,
                         random_number_seed=random_number_seed,
                         evaluation_type=evaluation_type)

    generator = Generator(MT19937(random_number_seed))
    sampling = Sampling(lower_bounds=par_lower_limit,
                        upper_bounds=par_upper_limit,
                        sample_properties=random_number_distribution,
                        generator=generator)

    parameter_guesses = sampling.generate_samples(
        number_of_samples=model.number_of_samples)  # Pre-allocate memory for the guesses

    resources.obj_func = obj_func
    resources.eq_func = eq_func
    resources.ineq_func = ineq_func

    if number_of_processes:
        resources.par_lower_limit = Array(c_double, model.par_lower_limit)
        resources.par_upper_limit = Array(c_double, model.par_upper_limit)
        resources.eq_values = Array(c_double, model.eq_values)
        resources.ineq_lower_bounds = Array(c_double, model.ineq_lower_bounds)
        resources.ineq_upper_bounds = Array(c_double, model.ineq_upper_bounds)
        resources.parameter_guesses = Array(c_double, parameter_guesses)
        resources.pysolnp_delta = Value(c_double, model.delta)
        resources.pysolnp_rho = Value(c_double, model.rho)
        resources.pysolnp_max_major_iter = Value(c_int, model.max_major_iter)
        resources.pysolnp_max_minor_iter = Value(c_int, model.max_minor_iter)
        resources.pysolnp_tolerance = Value(c_double, model.tolerance)
        resources.pysolnp_debug = Value(c_bool, model.debug)
        resources.evaluation_type = Value(c_int, model.evaluation_type)
        resources.number_of_parameters = Value(c_int, model.number_of_parameters)

        with multi.Pool(processes=number_of_processes) as pool:

            objective_results = pool.map(evaluate_starting_guess, range(model.number_of_samples))

            solutions = nsmallest(n=model.number_of_restarts, iterable=enumerate(objective_results),
                                  key=lambda value: value[1])
            solve_indices = [index for value, index in solutions]
            final_results = pool.map(pysolnp_solve, solve_indices)

            solutions = [solution for solution in final_results if model.check_solution_feasibility(solution)]

    else:
        resources.par_lower_limit = model.par_lower_limit
        resources.par_upper_limit = model.par_upper_limit
        resources.eq_values = model.eq_values
        resources.ineq_lower_bounds = model.ineq_lower_bounds
        resources.ineq_upper_bounds = model.ineq_upper_bounds
        resources.parameter_guesses = parameter_guesses
        resources.pysolnp_delta = model.delta
        resources.pysolnp_rho = model.rho
        resources.pysolnp_max_major_iter = model.max_major_iter
        resources.pysolnp_max_minor_iter = model.max_minor_iter
        resources.pysolnp_tolerance = model.tolerance
        resources.pysolnp_debug = model.debug
        resources.evaluation_type = model.evaluation_type
        resources.number_of_parameters = model.number_of_parameters

        objective_results = map(evaluate_starting_guess, range(model.number_of_samples))

        solutions = nsmallest(n=model.number_of_restarts, iterable=enumerate(objective_results),
                              key=lambda value: value[1])
        solve_indices = [index for index, value in solutions]
        final_results = map(pysolnp_solve, solve_indices)

        solutions = [solution for solution in final_results if model.check_solution_feasibility(solution)]

    # Finally, pysolnp might have not converged for any solution. Check if we have any feasible solution.
    if len(solutions) == 0:
        raise Exception(f"Not able to find any feasible solution in {number_of_restarts} restarts.")

    solutions = [(solution, obj_func(solution)) for solution in solutions]

    best_solution = reduce(lambda first, second: first if first[1] < second[1] else second, solutions)
    return best_solution
