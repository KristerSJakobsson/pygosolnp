from ctypes import c_long, c_bool, c_double, c_int
from typing import Any

import pysolnp

from pygosolnp import resources
from pygosolnp.model import EvaluationType


def initialize_worker_process_resources(obj_func,
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
                                        restart_results):
    """
    This function is used to provide the functions in this file access to shared resources when running pygosolnp.
     In multiprocess-mode it will pass multiprocess-safe types for each process, in single-processing it will simply have regular Python types.
    :param obj_func: A pickleable (global) callback objective function
    :param par_lower_limit: An double / multiprocessing.Array (float) representing parameter lower constraints
    :param par_upper_limit: An double / multiprocessing.Array (float) representing parameter upper constraints
    :param eq_func: [Optional, default None] A pickleable (global) callback equality function
    :param eq_values: [Optional, default None] An List / multiprocessing.Array (float) representing the equality constraints
    :param ineq_func: [Optional, default None] A pickleable (global) callback inequality function
    :param ineq_lower_bounds: [Optional, default None] An List / multiprocessing.Array (float) representing the inequality lower constraints
    :param ineq_upper_bounds: [Optional, default None] An List / multiprocessing.Array (float) representing the inequality upper constraints
    :param parameter_guesses: An List / multiprocessing.Array (float) representing randomly generated starting guesses
    :param pysolnp_delta: An double / multiprocessing.Value (float) representing pyolnp delta parameter
    :param pysolnp_rho: An double / multiprocessing.Value (float) representing pyolnp rho parameter
    :param pysolnp_max_major_iter: An int / multiprocessing.Value (int) representing pyolnp max major iterations parameter
    :param pysolnp_max_minor_iter: An int  / multiprocessing.Value (int) representing pyolnp max minor iterations parameter
    :param pysolnp_tolerance: An double / multiprocessing.Value (float) representing pyolnp tolerance parameter
    :param pysolnp_debug: An bool / multiprocessing.Value (bool) representing pyolnp debug parameter
    :param evaluation_type: An int / multiprocessing.Value (int) representing the EvaluationType enum mappings for pygosolnp
    :param number_of_parameters: An int / multiprocessing.Value (int) representing the number of parameters for this problem (a.k.a len(par_lower_limit))
    :param eval_results: An List / multiprocessing.Array (float) for storing the evaluation function results
    :param restart_results: An List / multiprocessing.Array (float) for storing the pysolnp calculation parameter results
    """
    resources.obj_func = obj_func
    resources.par_lower_limit = par_lower_limit
    resources.par_upper_limit = par_upper_limit
    resources.eq_func = eq_func
    resources.eq_values = eq_values
    resources.ineq_func = ineq_func
    resources.ineq_lower_bounds = ineq_lower_bounds
    resources.ineq_upper_bounds = ineq_upper_bounds
    resources.parameter_guesses = parameter_guesses
    resources.pysolnp_delta = pysolnp_delta
    resources.pysolnp_rho = pysolnp_rho
    resources.pysolnp_max_major_iter = pysolnp_max_major_iter
    resources.pysolnp_max_minor_iter = pysolnp_max_minor_iter
    resources.pysolnp_tolerance = pysolnp_tolerance
    resources.pysolnp_debug = pysolnp_debug
    resources.evaluation_type = evaluation_type
    resources.number_of_parameters = number_of_parameters
    resources.eval_results = eval_results
    resources.restart_results = restart_results


def __resource_value(resource: Any):
    type_of_value = type(resource)
    if type_of_value in [c_long, c_double, c_int, c_bool]:
        return resource.value
    return resource


def objective_func_exclude_ineq(variables, obj_func, eq_func, eq_values, ineq_func, ineq_lower_bounds,
                                ineq_upper_bounds):
    if ineq_func is not None:
        # Exclude any inequality violations by setting their value to infinity
        ineq_values = ineq_func(variables)
        is_outside_ineq_constraints = any(
            value < ineq_lower_bounds[index] or value > ineq_upper_bounds[index] for index, value in
            enumerate(ineq_values))
        if is_outside_ineq_constraints:
            return float("inf")

    if obj_func is None:
        raise ValueError(f"The objective function supplied could not be identified.")
    obj_value = obj_func(variables)
    return obj_value


def penalty_barrier_function(variables, obj_func, eq_func, eq_values, ineq_func, ineq_lower_bounds, ineq_upper_bounds):
    objective_result = obj_func(variables)
    if ineq_func is not None and ineq_upper_bounds is not None and ineq_lower_bounds is not None:
        def barrier_value_function(value: float):
            if value <= 0.0:
                return 0.0
            else:
                return (0.9 + value) ** 2

        inequality_values = ineq_func(variables)
        for index, value in enumerate(inequality_values):
            objective_result += 100.0 * (
                    barrier_value_function(ineq_lower_bounds[index] - value) +
                    barrier_value_function(value - ineq_upper_bounds[index])
            )

    if eq_func is not None and eq_values is not None:
        equality_values = eq_func(variables)
        objective_result += sum((value - eq_values[index]) ** 2 for index, value in enumerate(equality_values)) / 100.0

    return objective_result


def evaluate_starting_guess(simulation_index: int):
    guesses = __resource_value(resources.parameter_guesses)
    eval_type = __resource_value(resources.evaluation_type)
    number_of_parameters = __resource_value(resources.number_of_parameters)

    start_index = simulation_index * number_of_parameters
    end_index = (simulation_index + 1) * number_of_parameters
    try:
        eval_objective_function = {
            EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ.value: objective_func_exclude_ineq,
            EvaluationType.PENALTY_BARRIER_FUNCTION.value: penalty_barrier_function
        }

        eval_func = eval_objective_function[eval_type]
        eval_result = eval_func(variables=guesses[start_index: end_index],
                                obj_func=__resource_value(resources.obj_func),
                                eq_func=__resource_value(resources.eq_func),
                                eq_values=__resource_value(resources.eq_values),
                                ineq_func=__resource_value(resources.ineq_func),
                                ineq_lower_bounds=__resource_value(resources.ineq_lower_bounds),
                                ineq_upper_bounds=__resource_value(resources.ineq_upper_bounds))
        resources.eval_results[simulation_index] = eval_result
    except Exception as ex:
        resources.eval_results[simulation_index] = float("inf")


def pysolnp_solve(solve_index: int, guess_index: int):
    number_of_parameters = __resource_value(resources.number_of_parameters)
    debug = __resource_value(resources.pysolnp_debug)
    start_value = resources.parameter_guesses[(guess_index * number_of_parameters): (
            (guess_index + 1) * number_of_parameters)]

    try:
        solve_result: pysolnp.Result = pysolnp.solve(obj_func=__resource_value(resources.obj_func),
                                                     par_start_value=start_value,
                                                     par_lower_limit=__resource_value(resources.par_lower_limit),
                                                     par_upper_limit=__resource_value(resources.par_upper_limit),
                                                     eq_func=__resource_value(resources.eq_func),
                                                     eq_values=__resource_value(resources.eq_values),
                                                     ineq_func=__resource_value(resources.ineq_func),
                                                     ineq_lower_bounds=__resource_value(resources.ineq_lower_bounds),
                                                     ineq_upper_bounds=__resource_value(resources.ineq_upper_bounds),
                                                     rho=__resource_value(resources.pysolnp_rho),
                                                     max_major_iter=__resource_value(resources.pysolnp_max_major_iter),
                                                     max_minor_iter=__resource_value(resources.pysolnp_max_minor_iter),
                                                     delta=__resource_value(resources.pysolnp_delta),
                                                     tolerance=__resource_value(resources.pysolnp_tolerance),
                                                     debug=debug)

        resources.restart_results[(solve_index * number_of_parameters): (
                (solve_index + 1) * number_of_parameters)] = solve_result.optimum
    except ValueError as value_error:
        if debug:
            print(f"Error happened when running pysolnp for guess with index {guess_index}, ignoring this result. Error message: {value_error}")
