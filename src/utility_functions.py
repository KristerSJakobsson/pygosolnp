def lagrangian_function(x, obj_func, eq_func, eq_values, ineq_func, ineq_lower_bounds, ineq_upper_bounds):
    objective_result = obj_func(x)

    if ineq_func is not None and ineq_upper_bounds is not None and ineq_lower_bounds is not None:
        def barrier_value_function(value: float):
            if value <= 0.0:
                return 0.0
            else:
                return (0.9 + value) ** 2

        inequality_values = ineq_func(x)
        for index, value in enumerate(inequality_values):
            objective_result += 100.0 * (
                    barrier_value_function(ineq_lower_bounds[index] - value) +
                    barrier_value_function(value - ineq_upper_bounds[index])
            )

    if eq_func is not None and eq_values is not None:
        equality_values = eq_func(x)
        objective_result += sum(
            (equality_value - eq_values[index]) ** 2 for index, equality_value in enumerate(equality_values)) / 100.0

    return objective_result
