from collections import namedtuple
from ctypes import c_int, c_double, c_bool
from functools import reduce
from heapq import nsmallest
from multiprocessing import Array, Value, Pool
from typing import Callable, List, Optional, Iterable, Union

from numpy.random import Generator, MT19937

from src.evaluation_functions import evaluate_starting_guess, pysolnp_solve, initialize_worker_process_resources
from src.model import ProblemModel, EvaluationType
from src.sampling import Distribution, Sampling

Result = namedtuple(typename="Result", field_names=("parameters", "obj_value", "converged"))


class Results:
    def __init__(self, solutions: Iterable[Result], starting_guesses: List[float]):
        self.__solutions = solutions
        self.__starting_guesses = starting_guesses

    @property
    def all_solutions(self) -> Iterable[Result]:
        return self.__solutions

    @property
    def best_solution(self):
        viable_solutions = [solution for solution in self.__solutions if solution.converged]
        return reduce(lambda first, second: first if first.obj_value < second.obj_value else second, viable_solutions)

    @property
    def starting_guesses(self) -> Iterable[float]:
        return self.__starting_guesses


def get_best_solutions(results: Iterable, number_of_results: int):
    result = nsmallest(n=number_of_results,
                       iterable=enumerate(results),
                       key=lambda value: value[1])
    return result


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
          number_of_processes: Optional[int] = None,
          random_number_distribution: Optional[List[Distribution]] = None,
          random_number_seed: Optional[int] = None,
          evaluation_type: Union[EvaluationType, int] = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ) -> Results:
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
        number_of_samples=model.number_of_simulations)  # Pre-allocate memory for the guesses

    if number_of_processes:
        par_lower_limit = Array(c_double, model.par_lower_limit, lock=False)
        par_upper_limit = Array(c_double, model.par_upper_limit, lock=False)
        eq_values = Array(c_double, model.eq_values, lock=False) if model.has_eq_bounds else None
        ineq_lower_bounds = Array(c_double, model.ineq_lower_bounds, lock=False) if model.has_ineq_bounds else None
        ineq_upper_bounds = Array(c_double, model.ineq_upper_bounds, lock=False) if model.has_ineq_bounds else None
        pysolnp_delta = Value(c_double, model.delta, lock=False)
        pysolnp_rho = Value(c_double, model.rho, lock=False)
        pysolnp_max_major_iter = Value(c_int, model.max_major_iter, lock=False)
        pysolnp_max_minor_iter = Value(c_int, model.max_minor_iter, lock=False)
        pysolnp_tolerance = Value(c_double, model.tolerance, lock=False)
        pysolnp_debug = Value(c_bool, model.debug, lock=False)
        evaluation_type = Value(c_int, model.evaluation_type.value, lock=False)
        number_of_parameters = Value(c_int, model.number_of_parameters, lock=False)

        parameter_guesses = Array(c_double, parameter_guesses, lock=False)
        eval_results = Array(c_double, model.number_of_simulations)  # Results from the eval function
        restart_results = Array(c_double, model.number_of_simulations * model.number_of_parameters) # Results from pysolnp restarts

        initargs = (
            obj_func,
            par_lower_limit,
            par_upper_limit,
            eq_func,
            eq_values,
            ineq_func,
            ineq_lower_bounds,
            ineq_upper_bounds,
            parameter_guesses,
            pysolnp_delta,
            pysolnp_rho,
            pysolnp_max_major_iter,
            pysolnp_max_minor_iter,
            pysolnp_tolerance,
            pysolnp_debug,
            evaluation_type,
            number_of_parameters,
            eval_results,
            restart_results
        )
        with Pool(processes=number_of_processes,
                  initializer=initialize_worker_process_resources,
                  initargs=initargs) as pool:

            pool.map(evaluate_starting_guess, range(model.number_of_simulations))

            best_solutions = get_best_solutions(results=eval_results, number_of_results=model.number_of_restarts)
            solve_guess_indices = [index for index, value in best_solutions]
            # The found optimums are stored in parameter_guesses
            pool.starmap(pysolnp_solve, enumerate(solve_guess_indices))

    else:
        eval_results = [None] * model.number_of_simulations
        restart_results = [None] * model.number_of_simulations * model.number_of_restarts

        initialize_worker_process_resources(
            obj_func=obj_func,
            par_lower_limit=model.par_lower_limit,
            par_upper_limit=model.par_upper_limit,
            eq_func=eq_func if model.has_eq_bounds else None,
            eq_values=model.eq_values if model.has_eq_bounds else None,
            ineq_func=ineq_func if model.has_ineq_bounds else None,
            ineq_lower_bounds=model.ineq_lower_bounds if model.has_ineq_bounds else None,
            ineq_upper_bounds=model.ineq_upper_bounds if model.has_ineq_bounds else None,
            parameter_guesses=parameter_guesses,
            pysolnp_delta=model.delta,
            pysolnp_rho=model.rho,
            pysolnp_max_major_iter=model.max_major_iter,
            pysolnp_max_minor_iter=model.max_minor_iter,
            pysolnp_tolerance=model.tolerance,
            pysolnp_debug=model.debug,
            evaluation_type=model.evaluation_type.value,
            number_of_parameters=model.number_of_parameters,
            eval_results=eval_results,
            restart_results=restart_results
        )

        for index in range(model.number_of_simulations):
            evaluate_starting_guess(simulation_index=index)

        best_solutions = get_best_solutions(results=eval_results, number_of_results=model.number_of_restarts)
        solve_guess_indices = [index for index, value in best_solutions]
        # The found optimums are stored in parameter_guesses
        for solve_index, guess_index in enumerate(solve_guess_indices):
            pysolnp_solve(solve_index=solve_index, guess_index=guess_index)

    # For each restart, get the resulting parameters
    solutions = [restart_results[index * model.number_of_parameters: (index + 1) * model.number_of_parameters] for
                 index in range(model.number_of_restarts)]

    # Each Result represents a solution to the restart (might have not converged)
    best_solutions = [
        Result(parameters=solution, obj_value=obj_func(solution), converged=model.check_solution_feasibility(solution))
        for solution in solutions]

    # pysolnp might have not converged for some solution, if no converging solutions exist, print an warning message.
    if len([solution for solution in best_solutions if solution.converged]) == 0:
        print(f"Not able to find any feasible solution in {number_of_restarts} restarts.")

    return Results(solutions=best_solutions, starting_guesses=parameter_guesses)
