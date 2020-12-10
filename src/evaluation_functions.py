from typing import List

import pysolnp

from src import resources
from src.model import EvaluationType


def objective_func_exclude_ineq(variables, obj_func, eq_func, eq_values, ineq_func, ineq_lower_bounds,
                                ineq_upper_bounds):
    if ineq_func is not None:
        # Exclude any inequality violations by setting their value to infinity
        ineq_values = ineq_func(variables)
        has_ineq_validation = any(
            value < ineq_lower_bounds or value > ineq_upper_bounds for value in ineq_values)
        if has_ineq_validation:
            return float("inf")

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
        objective_result += sum(
            (equality_value - eq_values[index]) ** 2 for index, equality_value in
            enumerate(equality_values)) / 100.0

    return objective_result


def evaluate_starting_guess(simulation_index) -> float:
    eval_value = resources.parameter_guesses[(simulation_index * resources.number_of_parameters): (
            (simulation_index + 1) * resources.number_of_parameters)]
    try:
        eval_objective_function = {
            EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ.value: objective_func_exclude_ineq,
            EvaluationType.PENALTY_BARRIER_FUNCTION.value: penalty_barrier_function
        }

        eval_func = eval_objective_function[resources.evaluation_type.value]
        objective_result = eval_func(variables=eval_value,
                                     obj_func=resources.obj_func,
                                     eq_func=resources.eq_func,
                                     eq_values=resources.eq_values,
                                     ineq_func=resources.ineq_func,
                                     ineq_lower_bounds=resources.ineq_lower_bounds,
                                     ineq_upper_bounds=resources.ineq_upper_bounds)
        return objective_result
    except Exception as ex:
        return float("inf")


def pysolnp_solve(simulation_index) -> List[float]:
    start_value = resources.parameter_guesses[(simulation_index * resources.number_of_parameters): (
            (simulation_index + 1) * resources.number_of_parameters)]

    solve_result = pysolnp.solve(obj_func=resources.obj_func,
                                 par_start_value=start_value,
                                 par_lower_limit=resources.par_lower_limit,
                                 par_upper_limit=resources.par_upper_limit,
                                 eq_func=resources.eq_func,
                                 eq_values=resources.eq_values,
                                 ineq_func=resources.ineq_func,
                                 ineq_lower_bounds=resources.ineq_lower_bounds,
                                 ineq_upper_bounds=resources.ineq_upper_bounds,
                                 rho=resources.pysolnp_rho,
                                 max_major_iter=resources.pysolnp_max_major_iter,
                                 max_minor_iter=resources.pysolnp_max_minor_iter,
                                 delta=resources.pysolnp_delta,
                                 tolerance=resources.pysolnp_tolerance,
                                 debug=resources.pysolnp_debug)

    return list(solve_result.optimum)
