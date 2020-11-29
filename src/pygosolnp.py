from random import Random
import pysolnp
from functools import reduce
import multiprocessing as multi

from typing import Callable, List, Optional
from enum import Enum


class Distribution(Enum):
    UNIFORM = "uniform" # No confidence in the location of the parameter
    # TODO: Add support for the below
    # TRUNCATED_NORMAL = "truncnorm" # High confidence in the location of the parameter
    # NORMAL = "normal" # Uncertainty in Lower and Upper bounds, but some idea about the dispersion about the location

class EvaluationType(Enum):
    OBJECTIVE_FUNC_EXCLUDE_INEQ = 1 # Exclude any guesses that violate the objective function and the evaluate with objective function
    # PENALTY_BARRIER_FUNCTION    = 2  # Use the penalty barrier function to evaluate objective function


def _generate_starting_parameters(generator: Random,
                                  obj_func: Callable,
                                  par_lower_limit: List[float],
                                  par_upper_limit: List[float],
                                  number_of_restarts: int,
                                  number_of_simulations: int,
                                  par_start_value: Optional[List[float]],
                                  eq_func: Optional[Callable],
                                  eq_values: Optional[List[float]],
                                  ineq_func: Optional[Callable],
                                  ineq_lower_bounds: Optional[List[float]],
                                  ineq_upper_bounds: Optional[List[float]],
                                  fixed_starting_parameters: Optional[List[float]],
                                  evaluation_type: EvaluationType) -> List[Optional[float]]:
    """
    Generate starting parameters.
    1. Generate `number_of_simulations` * `number_of_restarts` sets of starting parameters
    2. See if any of them violate inequality constraints, if they do, then remove those starting parameters
    3. Calculate the objective function for each starting parameter and store the best `number_of_restarts` starting parameters
    :param generator: The Random Number Generator used for getting the starting parameters
    :param ... the problem definition
    :return: A list of the best `number_of_restarts` starting parameters
    """

    # Step 1. Generate starting parameters
    number_of_parameters = len(par_lower_limit)
    parameter_guesses: List[Optional[float]] = [None] * (number_of_restarts * number_of_simulations * number_of_parameters) # Pre-allocate memory for the guesses
    objective_results: List[Optional[float]] = [None] * (number_of_restarts * number_of_simulations)  # Pre-allocate memory for the objective function values

    for simulation_index in range(number_of_restarts * number_of_simulations):
        for parameter_index in range(number_of_parameters):
            parameter_guesses[number_of_parameters * simulation_index + parameter_index] = generator.uniform(par_lower_limit[simulation_index], par_upper_limit[simulation_index])

    # Step 2. Evaluate inequality constraints
    if ineq_func:
        for simulation_index in range(number_of_restarts * number_of_simulations):
            try:
                ineq_values = ineq_func(parameter_guesses[(number_of_parameters * simulation_index): (number_of_parameters * (simulation_index + 1) - 1)])
                has_ineq_validation = any(value < ineq_lower_bounds or value > ineq_upper_bounds for value in ineq_values)
                if has_ineq_validation:
                    objective_results[simulation_index] = float("inf")
            except Exception as ex:
                objective_results[simulation_index] = float("inf")

    # Step 3. Evaluate objective function
    for simulation_index in range(number_of_restarts * number_of_simulations):
        if objective_results[number_of_parameters * simulation_index] != float("inf"):
            try:
                obj_value = obj_func(parameter_guesses[(number_of_parameters * simulation_index): (number_of_parameters * (simulation_index + 1) - 1)])
                objective_results[simulation_index] = obj_value
            except Exception as ex:
                objective_results[simulation_index] = float("inf")

    result = sorted(objective_results)
    return result[:number_of_restarts]

def solve(obj_func: Callable,
          par_lower_limit: List[float],
          par_upper_limit: List[float],
          number_of_restarts: int,
          number_of_simulations: int,
          par_start_value: Optional[List[float]] = None,
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
          random_number_seed: Optional[int] = None,
          evaluation_type: EvaluationType = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ) -> pysolnp.Result:

    if type(par_lower_limit) is not List or type(par_upper_limit) is not List:
        raise ValueError("Parameter lower and upper bounds are required for gosolnp")

    if len(par_lower_limit) != len(par_upper_limit):
        raise ValueError("The length of lower and upper bounds are not the same")

    # Expand fixed starting parameters if any
    actual_obj_func = obj_func
    actual_eq_func = eq_func
    actual_ineq_func = ineq_func
    if fixed_starting_parameters is not None:
        raise Exception("Not yet supported...")
        # TODO: Add this support
        # def modify_parameteres(x: List[float]) -> List[float]:
        #     original_index = 0
        #     new_x = fixed_starting_parameters.copy()
        #     for index in range(len(new_x)):
        #         if new_x[index] is None:
        #             new_x[index] = x[original_index]
        #             original_index = original_index + 1
        #     return new_x
        #
        # actual_obj_func = lambda x : obj_func(modify_parameteres(x))
        #
        # if eq_func is not None:
        #     actual_eq_func = lambda x: obj_func(modify_parameteres(x))
        #
        # if ineq_func is not None:
        #     actual_ineq_func = lambda x: obj_func(modify_parameteres(x))

    if random_number_seed is None:
        random_number_generator = Random()
    else:
        random_number_generator = Random(random_number_seed)

    starting_parameters = _generate_starting_parameters(generator=random_number_generator,
                                                        obj_func=obj_func,
                                                        par_lower_limit=par_lower_limit,
                                                        par_upper_limit=par_upper_limit,
                                                        number_of_restarts=number_of_restarts,
                                                        number_of_simulations=number_of_simulations,
                                                        par_start_value=par_start_value,
                                                        eq_func=eq_func,
                                                        eq_values=eq_func,
                                                        ineq_func=ineq_func,
                                                        ineq_lower_bounds=ineq_lower_bounds,
                                                        ineq_upper_bounds=ineq_upper_bounds,
                                                        fixed_starting_parameters=fixed_starting_parameters,
                                                        evaluation_type=evaluation_type)

    solutions = []
    if number_of_processes is None:
        # Run single-processed
        for restart_index in range(number_of_restarts):
            solve_result = pysolnp.solve(obj_func=actual_obj_func,
                          par_start_value=starting_parameters[restart_index],
                          par_lower_limit=par_lower_limit,
                          par_upper_limit=par_upper_limit,
                          eq_func=actual_eq_func,
                          eq_values=eq_values,
                          ineq_func=actual_ineq_func,
                          ineq_lower_bounds=ineq_lower_bounds,
                          ineq_upper_bounds=ineq_upper_bounds,
                          rho=rho,
                          max_major_iter=max_major_iter,
                          max_minor_iter=max_minor_iter,
                          delta=delta,
                          tolerance=tolerance,
                          debug=debug)

            solutions.append(solve_result)
    else:
        # TODO: Add support for multiprocessing
        # pool = multi.Pool(processes=number_of_processes)
        pass

    best_solution = reduce(lambda first, second: first if first.solve_value < second.solve_value else second, solutions)
    return best_solution
