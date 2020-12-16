# Use the alkyla function from pysolnp benchmarks to tests lagrangian function
def alkyla_objective_function(x):
    result = -0.63 * x[3] * x[6] + 50.4 * x[0] + 3.5 * x[1] + x[2] + 33.6 * x[4]
    return result


def alkyla_equality_function(x):
    result = [
        98.0 * x[2] - 0.1 * x[3] * x[5] * x[8] - x[2] * x[5],
        1000.0 * x[1] + 100.0 * x[4] - 100.0 * x[0] * x[7],
        122.0 * x[3] - 100.0 * x[0] - 100.0 * x[4],
    ]
    return result


def alkyla_inequality_function(x):
    result = [
        (1.12 * x[0] + 0.13167 * x[0] * x[7] - 0.00667 * x[0] * x[7] * x[7]) / x[3],
        (1.098 * x[7] - 0.038 * x[7] * x[7] + 0.325 * x[5] + 57.25) / x[6],
        (-0.222 * x[9] + 35.82) / x[8],
        (3.0 * x[6] - 133.0) / x[9],
    ]
    return result


equality_values = [0, 0, 0]
inequality_lower_bounds = [0.99, 0.99, 0.9, 0.99]
inequality_upper_bounds = [100 / 99, 100 / 99, 10 / 9, 100 / 99]
parameter_lower_bounds = [0.0, 0.0, 0.0, 10.0, 0.0, 85.0, 10.0, 3.0, 1.0, 145.0]
parameter_upper_bounds = [20.0, 16.0, 120.0, 50.0, 20.0, 93.0, 95.0, 12.0, 4.0, 162.0]
