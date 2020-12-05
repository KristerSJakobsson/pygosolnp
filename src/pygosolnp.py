from random import Random
import pysolnp
from functools import reduce
import multiprocessing as multi
from src.utility_functions import lagrangian_function

from typing import Callable, List, Optional, Tuple
from enum import Enum


class Distribution(Enum):
    UNIFORM = "uniform"  # No confidence in the location of the parameter
    # TODO: Add support for the below
    # TRUNCATED_NORMAL = "truncnorm" # High confidence in the location of the parameter
    # NORMAL = "normal" # Uncertainty in Lower and Upper bounds, but some idea about the dispersion about the location


class EvaluationType(Enum):
    OBJECTIVE_FUNC_EXCLUDE_INEQ = 1  # Exclude any guesses that violate the objective function and the evaluate with objective function
    PENALTY_BARRIER_FUNCTION = 2  # Use the penalty barrier function to evaluate objective function


def obj_function_exclude_inequalities(
        generator: Random,
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
    pass


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
    2. Calculate the objective function for each starting parameter (based on eval. type) and store the best `number_of_restarts` starting parameters
    :param generator: The Random Number Generator used for getting the starting parameters
    :param ... the problem definition
    :return: A list of the best `number_of_restarts` starting parameters
    """

    # Step 1. Generate starting parameters
    number_of_parameters = len(par_lower_limit)
    parameter_guesses: List[Optional[float]] = [None] * (
            number_of_restarts * number_of_simulations * number_of_parameters)  # Pre-allocate memory for the guesses
    objective_results: List[Optional[Tuple[float, List]]] = [None] * (
            number_of_restarts * number_of_simulations)  # Pre-allocate memory for the objective function values

    for simulation_index in range(number_of_restarts * number_of_simulations):
        for parameter_index in range(number_of_parameters):
            parameter_guesses[parameter_index + number_of_parameters * simulation_index] = generator.uniform(
                par_lower_limit[parameter_index], par_upper_limit[parameter_index])

    def objective_func_exclude_ineq():
        for simulation_index in range(number_of_restarts * number_of_simulations):
            variables = parameter_guesses[(simulation_index * number_of_parameters): ((simulation_index + 1) * number_of_parameters)]
            if ineq_func is not None:
                # Exclude any inequality violations by setting their value to infinity
                try:
                    ineq_values = ineq_func(variables)
                    has_ineq_validation = any(
                        value < ineq_lower_bounds or value > ineq_upper_bounds for value in ineq_values)
                    if has_ineq_validation:
                        objective_results[simulation_index] = (float("inf"), variables)
                        continue
                except Exception as ex:
                    objective_results[simulation_index] = (float("inf"), variables)
                    continue

            try:
                obj_value = obj_func(variables)
                objective_results[simulation_index] = (obj_value, variables)
            except Exception as ex:
                objective_results[simulation_index] = (float("inf"), variables)

    def penalty_barrier_function():

        for simulation_index in range(number_of_restarts * number_of_simulations):
            variables = parameter_guesses[(simulation_index * number_of_parameters): (
                    (simulation_index + 1) * number_of_parameters)]
            try:
                objective_result = lagrangian_function(x=variables,
                                                       obj_func=obj_func,
                                                       eq_func=eq_func,
                                                       eq_values=eq_values, ineq_func=ineq_func,
                                                       ineq_lower_bounds=ineq_lower_bounds,
                                                       ineq_upper_bounds=ineq_upper_bounds)
                objective_results[simulation_index] = (objective_result, variables)
            except Exception as ex:
                objective_results[simulation_index] = (float("inf"), variables)

    eval_objective_function = {
        EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ: objective_func_exclude_ineq,
        EvaluationType.PENALTY_BARRIER_FUNCTION: penalty_barrier_function
    }

    eval_objective_function[evaluation_type]()
    result = sorted(objective_results, key=lambda value: value[0])
    return [value[1] for value in result[:number_of_restarts]]


def check_solution_feasibility(par_found_solution,
                               par_lower_limit,
                               par_upper_limit,
                               eq_func,
                               eq_values,
                               ineq_func,
                               ineq_lower_bounds,
                               ineq_upper_bounds,
                               tolerance):
    if any(value < par_lower_limit[index] - tolerance or par_upper_limit[index] + tolerance < value for index, value in
           enumerate(par_found_solution)):
        return False

    if eq_func is not None:
        equality_function_values = eq_func(par_found_solution)
        if any(value < eq_values[index] - tolerance or eq_values[index] + tolerance < value for index, value in
               enumerate(equality_function_values)):
            return False

    if ineq_func is not None:
        inequality_function_values = ineq_func(par_found_solution)
        if any(value < ineq_lower_bounds[index] - tolerance or ineq_upper_bounds[index] + tolerance < value for
               index, value in enumerate(inequality_function_values)):
            return False

    return True


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
    if type(par_lower_limit) is not list or type(par_upper_limit) is not list:
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
                                                        obj_func=actual_obj_func,
                                                        par_lower_limit=par_lower_limit,
                                                        par_upper_limit=par_upper_limit,
                                                        number_of_restarts=number_of_restarts,
                                                        number_of_simulations=number_of_simulations,
                                                        par_start_value=par_start_value,
                                                        eq_func=actual_eq_func,
                                                        eq_values=eq_values,
                                                        ineq_func=actual_ineq_func,
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

            if check_solution_feasibility(par_found_solution=solve_result.optimum,
                                          par_lower_limit=par_lower_limit,
                                          par_upper_limit=par_upper_limit,
                                          eq_func=actual_eq_func,
                                          eq_values=eq_values,
                                          ineq_func=actual_ineq_func,
                                          ineq_lower_bounds=ineq_lower_bounds,
                                          ineq_upper_bounds=ineq_upper_bounds,
                                          tolerance=tolerance):
                solutions.append(solve_result)
    else:
        # TODO: Add support for multiprocessing
        # pool = multi.Pool(processes=number_of_processes)
        pass

    # Finally, pysolnp might have not converged for any solution. Check if we have any feasible solution.
    if len(solutions) == 0:
        raise Exception(f"Not able to find any feasible solution in {number_of_restarts} restarts.")

    best_solution = reduce(lambda first, second: first if first.solve_value < second.solve_value else second, solutions)
    return best_solution
