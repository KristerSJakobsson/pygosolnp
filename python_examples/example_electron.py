import pygosolnp
from math import sqrt

number_of_charges = 25


def obj_func(data):
    x = data[0:number_of_charges]
    y = data[number_of_charges:2 * number_of_charges]
    z = data[2 * number_of_charges:3 * number_of_charges]

    result = 0.0
    for i in range(0, number_of_charges - 1):
        for j in range(i + 1, number_of_charges):
            result += 1.0 / sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)

    return result


def eq_func(data):
    x = data[0:number_of_charges]
    y = data[number_of_charges:2 * number_of_charges]
    z = data[2 * number_of_charges:3 * number_of_charges]
    result = [None] * number_of_charges
    for i in range(0, number_of_charges):
        result[i] = x[i] ** 2 + y[i] ** 2 + z[i] ** 2

    return result


parameter_lower_bounds = [-1] * number_of_charges * 3
parameter_upper_bounds = [1] * number_of_charges * 3

equality_constraints = [1] * number_of_charges

results = pygosolnp.solve(
    obj_func=obj_func,
    eq_func=eq_func,
    eq_values=equality_constraints,
    par_lower_limit=parameter_lower_bounds,
    par_upper_limit=parameter_upper_bounds,
    number_of_restarts=4,
    number_of_simulations=20000,
    number_of_processes=None,
    random_number_seed=443,
    max_major_iter=100,
    evaluation_type=pygosolnp.EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
    debug=False)

print(results.all_solutions)
print(results.best_solution)
