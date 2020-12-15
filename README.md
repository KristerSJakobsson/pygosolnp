WORK IN PROGRESS
# pygosolnp - Random initialization and multiple restarts using pysolnp
See full documentation on [http://solnp.readthedocs.io](https://solnp.readthedocs.io/en/latest/).


## Description
SOLNP solves the general nonlinear optimization problem on the form:
```
    minimize f(x)
      subject to
       g(x) = e_x
   l_h <= h(x) <= u_h
   l_x <   x   < u_X
```
where f(x), g(x) and h(x) are smooth functions.
SOLNP guarantees to find an local optimum, but not the global one.
GOSOLNP tries to find the global optimum in two steps:
1. Generate `n` random starting parameters based on some specified distribution and evaluate them based on one of two evaluation functions:
   1. Objective function `f(x)` for all `x` that satisfies the inequalit constraint `l_h <= h(x) <= u_h`
   2. Penalty function: ``
2. For the `m` starting parameters with the lowest evaluation function value, run pysolnp to find nearest optimum. 

## Compatability
Python Wheels are available for CPython:
- Windows: Python 2.7, 3.6+
- Linux: Python 2.7, 3.5+
- Mac OS: Python 2.7, 3.5+

Depends on pysolnp and numpy.
Note: For best results, building pysolnp from source is recommended, as BLAS and LAPACK will make a difference.

## Installation
Simply install the package through PyPi with:
`pip install pygosolnp`
<br>WARNING: WORK IN PROGRESS, <b>not yet avilable</b>

## Usage
Below is the Electron example, for the complete example see [/python_examples/example_electron.py](/python_examples/example_electron.py).
```python
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
```

Output:
```
```

## Parameters
The basic signature is:
```python
solve(obj_func: function, par_start_value: List, par_lower_limit: object = None, par_upper_limit: object = None, eq_func: object = None, eq_values: object = None, ineq_func: object = None, ineq_lower_bounds: object = None, ineq_upper_bounds: object = None, rho: float = 1.0, max_major_iter: int = 10, max_minor_iter: int = 10, delta: float = 1e-05, tolerance: float = 0.0001, debug: bool = False) -> pysolnp.Result
```

Inputs:

| Parameter          | Type                      | Default value*   | Description                                                                       |
| -------------------|:--------------------------|:-----------------|-----------------------------------------------------------------------------------|
| obj_func           | Callable\[List, float\]   | -                | The objective function f(x) to minimize.                                          |
| par_start_value    | List                      | -                | The starting parameter x_0.                                                       |
| par_lower_limit    | List                      | None             | The parameter lower limit x_l.                                                    |
| par_upper_limit    | List                      | None             | The parameter upper limit x_u.                                                    |
| eq_func            | Callable\[List, float\]   | None             | The equality constraint function h(x).                                            |
| eq_values          | List                      | None             | The equality constraint values e_x.                                               |
| ineq_func          | Callable\[List, float\]   | None             | The inequality constraint function g(x).                                          |
| ineq_lower_bounds  | List                      | None             | The inequality constraint lower limit g_l.                                        |
| ineq_upper_bounds  | List                      | None             | The inequality constraint upper limit g_l.                                        |
| rho                | float                     | 1.0              | Penalty weighting scalar for infeasability in the augmented objective function.** |
| max_major_iter     | int                       | 400              | Maximum number of outer iterations.                                               |
| max_minor_iter     | int                       | 800              | Maximum number of inner iterations.                                               |
| delta              | float                     | 1e-07            | Step-size for forward differentiation.                                            |
| tolerance          | float                     | 1e-08            | Relative tolerance on optimality.                                                 |
| debug              | bool                      | False            | If set to true some debug output will be printed.                                 |

*Defaults for configuration parameters are based on the defaults for Rsolnp.<br>
**Higher values means the solution will bring the solution into the feasible region with higher weight. Very high values might lead to numerical ill conditioning or slow down convergence.

Output:
The function returns the `pygosolnp.Results` with the below properties.

| Property           | Type           | Description                                           |
| -------------------|:---------------|-------------------------------------------------------|
| solve_value        | float          | The value of the objective function at optimum f(x*). |
| optimum            | List\[float\]  | A list of parameters for the optimum x*.              |
| callbacks          | int            | Number of callbacks done to find this optimum.        |

## Authors

* **Krister S Jakobsson** - *Implementation* - krister.s.jakobsson@gmail.com

## License

This project is licensed under the Boost License - see the [license](LICENSE.md) file for details.

## Acknowledgments

* **Yinyu Ye** -  Publisher and mastermind behind the original SOLNP algorithm,
[Original Sources](https://web.stanford.edu/~yyye/matlab/)
* **Alexios Ghalanos and Stefan Theussl** - The people behind RSOLNP and GOSOLNP,
