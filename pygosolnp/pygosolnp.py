from collections import namedtuple
from ctypes import c_int, c_double, c_bool
from functools import reduce
from heapq import nsmallest
from multiprocessing import Array, Value, Pool
from typing import Callable, Optional, Union, List

from pygosolnp.evaluation_functions import evaluate_starting_guess, pysolnp_solve, initialize_worker_process_resources
from pygosolnp.model import ProblemModel, EvaluationType
from pygosolnp.sampling import Distribution, Sampling, DefaultSampling

Result = namedtuple(typename="Result", field_names=("parameters", "obj_value", "converged"))


class Results:
    def __init__(self, results: List[Result], starting_guesses: List[float]):
        self.__results = results
        self.__starting_guesses = starting_guesses

    def __str__(self):
        return f"Results(all_results={self.all_results}, best_solution={self.best_solution}, starting_guesses={self.starting_guesses})"

    @property
    def all_results(self) -> List[Result]:
        return self.__results

    @property
    def best_solution(self) -> Optional[Result]:
        viable_solutions = [solution for solution in self.__results if solution.converged]
        if len(viable_solutions) == 0:
            return None
        return reduce(lambda first, second: first if first.obj_value < second.obj_value else second, viable_solutions)

    @property
    def starting_guesses(self) -> List[float]:
        return self.__starting_guesses


def __get_best_solutions(results: Union[List, Array], number_of_results: int):
    results = nsmallest(n=number_of_results,
                        iterable=enumerate(results),
                        key=lambda value: value[1])

    if all(result[1] == float("inf") for result in results):
        raise ValueError("Evaluation functions could not locate any successful starting guesses.")

    return results


def __debug_message_eval_functions(model: ProblemModel, eval_results: Union[Array, List]):
    number_of_failed_evaluations = sum((1 if value == float("inf") else 0) for value in eval_results)
    print(
        f"Out of {model.number_of_evaluations} evaluations {number_of_failed_evaluations} failed or returned infinity for evaluation function {model.evaluation_type.name}. Check for issues with your problem definition or try changing the evaluation function.")


def solve(obj_func: Callable,
          par_lower_limit: List[float],
          par_upper_limit: List[float],
          eq_func: Optional[Callable] = None,
          eq_values: Optional[List[float]] = None,
          ineq_func: Optional[Callable] = None,
          ineq_lower_bounds: Optional[List[float]] = None,
          ineq_upper_bounds: Optional[List[float]] = None,
          number_of_restarts: int = 1,
          number_of_simulations: int = 20000,
          number_of_processes: Optional[int] = None,
          start_guess_sampling: Union[None, List[Distribution], Sampling] = None,
          seed: Union[None, int] = None,
          evaluation_type: Union[EvaluationType, int] = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
          pysolnp_rho: float = 1.0,
          pysolnp_max_major_iter: int = 10,
          pysolnp_max_minor_iter: int = 10,
          pysolnp_delta: float = 1e-05,
          pysolnp_tolerance: float = 0.0001,
          debug: bool = False) -> Results:
    # Represent the problem with the below object
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
                         rho=pysolnp_rho,
                         max_major_iter=pysolnp_max_major_iter,
                         max_minor_iter=pysolnp_max_minor_iter,
                         delta=pysolnp_delta,
                         tolerance=pysolnp_tolerance,
                         debug=debug,
                         number_of_processes=number_of_processes,
                         start_guess_sampling=start_guess_sampling,
                         evaluation_type=evaluation_type)

    # Validate the inputs for the problem model
    model.validate()

    if start_guess_sampling is None or type(start_guess_sampling) is list:
        # Generate samples using the DefaultSampling object
        sampling = DefaultSampling(parameter_lower_bounds=par_lower_limit,
                                   parameter_upper_bounds=par_upper_limit,
                                   sample_properties=start_guess_sampling,
                                   seed=seed)
    elif isinstance(start_guess_sampling, Sampling):
        if seed is not None and debug is True:
            print(f"Warning: Seed value {seed} ignored due to user sampling override")
        # User provided Sampling instance
        sampling = start_guess_sampling
    else:
        raise ValueError(
            f"Provided parameter start_guess_sampling was not of expected type. Expected None, List[Distribution] or Sampling.")

    parameter_guesses = sampling.generate_all_samples(
        number_of_samples=model.number_of_evaluations,
        sample_size=model.sample_size)

    if debug is True:
        if any(guess is None for guess in parameter_guesses):
            print(f"Some of the random samples provided failed to generate, is your Sampling class setup correctly?")

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
        eval_results = Array(c_double, model.number_of_evaluations)  # Results from the eval function
        restart_results = Array(c_double,
                                model.number_of_restarts * model.number_of_parameters)  # Results from pysolnp restarts

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

            pool.map(evaluate_starting_guess, range(model.number_of_evaluations))

            if debug is True:
                __debug_message_eval_functions(model=model, eval_results=eval_results)

            best_evaluations = __get_best_solutions(results=eval_results, number_of_results=model.number_of_restarts)
            solve_guess_indices = [index for index, value in best_evaluations]
            # The found optimums are stored in parameter_guesses
            pool.starmap(pysolnp_solve, enumerate(solve_guess_indices))

    else:
        eval_results = [None] * model.number_of_evaluations
        restart_results = [None] * model.number_of_restarts * model.number_of_parameters

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

        for index in range(model.number_of_evaluations):
            evaluate_starting_guess(simulation_index=index)

        if debug is True:
            __debug_message_eval_functions(model=model, eval_results=eval_results)

        best_evaluations = __get_best_solutions(results=eval_results, number_of_results=model.number_of_restarts)
        solve_guess_indices = [index for index, value in best_evaluations]
        # The found optimums are stored in parameter_guesses
        for solve_index, guess_index in enumerate(solve_guess_indices):
            pysolnp_solve(solve_index=solve_index, guess_index=guess_index)

    # For each restart, get the resulting parameters
    solutions = [restart_results[index * model.number_of_parameters: (index + 1) * model.number_of_parameters] for
                 index in range(model.number_of_restarts)]

    # Each Result represents a solution to the restart (might have not converged)
    all_results = [
        Result(parameters=solution, obj_value=obj_func(solution), converged=model.check_solution_feasibility(solution))
        for solution in solutions]

    # pysolnp might have not converged for some solution, if no converging solutions exist, print an warning message.
    if len([solution for solution in all_results if solution.converged]) == 0:
        print(f"Not able to find any feasible solution in {number_of_restarts} restarts.")

    return Results(results=all_results, starting_guesses=parameter_guesses)
