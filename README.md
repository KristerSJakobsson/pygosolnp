[![codecov](https://codecov.io/gh/KristerSJakobsson/pygosolnp/branch/main/graph/badge.svg?token=49IZH9VPJI)](https://codecov.io/gh/KristerSJakobsson/pygosolnp)
[![Documentation Status](https://readthedocs.org/projects/solnp/badge/?version=latest)](https://solnp.readthedocs.io/en/latest/?badge=latest)

# pygosolnp - Random initialization and multiple restarts using pysolnp
See full documentation on [http://solnp.readthedocs.io](https://solnp.readthedocs.io/en/latest/).


## Description
GOSOLNP tries to find the optimum for the general nonlinear optimization problem on the form:
```
    minimize f(x)
      subject to
       g(x) = e_x
   l_h <= h(x) <= u_h
   l_x <   x   < u_X
```
where f(x), g(x) and h(x) are smooth functions.

GOSOLNP tries to find the global optimum for given problem as explained below:
1. Generate `n` random starting parameters based on some specified distribution and evaluate them (lower value is better) based on one of two evaluation functions::
   1. Objective function `f(x)` for all `x` that satisfies the inequalit constraint `l_h <= h(x) <= u_h`
   2. Penalty function: `f(x) + 100 * sum(max(0, 0.9 + l_x - g(x))^2 + max(0, 0.9 + g(x) - u_x)^2) + sum(h(x) - e_x)^2/100`
2. For the `m` starting parameters with the lowest evaluation function value, run pysolnp to find nearest optimum. 
3. Return the best valid solution among the ones found through the various starting parameters (lowest solution value within bounds)

## Compatability
Python source code written to be compatible with Python 3.6+.
Depends on the `pysolnp` library.
Note: `pysolnp` is available on pip but for best results building `pysolnp` from source is recommended, as BLAS and LAPACK will make a difference.

## Installation
Simply install the package through PyPi with:
`pip install pygosolnp`

## Usage
Below is the Electron example, for the complete example see [/python_examples/example_electron.py](https://github.com/KristerSJakobsson/pygosolnp/blob/main/python_examples/example_electron.py).
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
    number_of_simulations=20000,  # This represents the number of starting guesses to use
    evaluation_type=pygosolnp.EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,  # This specifies how starting guesses are evaluated
    number_of_restarts=4,  # This specifies how many restarts to run from the best starting guesses
    number_of_processes=None,  # None here means to run everything single-processed
    seed=443,  # Seed for reproducibility, if omitted the default random seed is used (typically cpu clock based)
    pysolnp_max_major_iter=100,  # Pysolnp property
    debug=False)

all_results = results.all_results
print("; ".join([f"Solution {index + 1}: {solution.obj_value}" for index, solution in enumerate(all_results)]))
best_solution = results.best_solution
print(f"Best solution {best_solution.obj_value} for parameters {best_solution.parameters}.")
```

Output:
```
Solution 1: 244.1550118432253; Solution 2: 243.9490050190484; Solution 3: 185.78533081425041; Solution 4: 244.07921194485854
Best solution 243.9490050190484 for parameters [0.34027682232302764, 0.6883848066130182, 0.40606935432390506, -0.48792021292031806, -0.9178828953524689, -0.8589108634903266, 0.5283358549116118, 0.5728961925249723, 0.050290270369804546, 0.2822196996653568, -0.28946049710390886, 0.9330667664325792, -0.417772874000437, -0.03124740841970295, -0.29956912974747735, -0.10795596769157587, 0.3549207051381202, -0.8488364868994906, -0.6188824315686104, 0.8670714826307561, 0.3619506513550691, -0.8251195998826993, 0.8981487824398298, -0.3070816517072349, -0.2904911409773652, -0.35929970112105275, 0.38265416704984406, 0.33719255494620365, 0.650145631465414, -0.009286462818493796, -0.465918386592264, -0.8033631014752087, -0.7401045643478271, 0.3960071831167597, 0.8935914355529017, -0.06721625611418029, -0.17225237197258644, -0.15098042850508767, -0.7478725125678873, -0.6812276561168169, -0.9904585930824136, 0.856850019939644, 0.5254197207147568, -0.7013999163528392, 0.48657417413232107, -0.26408411581924884, 0.07567634864162288, -0.11182932375860347, 0.6420701581875298, 0.9557533156823153, 0.8689739558884132, -0.6161997929208027, 0.8493580035603775, -0.5824473351785862, -0.396742846647288, 0.21258614114742239, -0.27471641138639197, 0.3521864803783331, -0.9168692752182334, -0.3490651393423451, 0.9548271150556085, -0.3157777175001705, -0.8959190430352647, -0.6631065760657502, 0.667972799116207, 0.08565844771052915, 0.3739510687437103, 0.058402195224914855, -0.3535858335621826, 0.10692372605836734, -0.8940086985713928, 0.5598670317503106, 0.4252328966724537, 0.7024576680631351, 0.04637407503369189].
```

## Parameters
The basic signature is:
```python
pygosolnp.solve(
          obj_func: Callable,
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
          debug: bool = False) -> Results
```

Inputs:

| Parameter                  | Type                             | Default value*                             | Description                                                                                                                                |
| ---------------------------|:---------------------------------|:-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| obj_func                   | Callable\[List\[float\]\]        | -                                          | The objective function f(x) to minimize.                                                                                                   |
| par_lower_limit            | List\[float\]                    | -                                          | The parameter lower limit x_l.                                                                                                             |
| par_upper_limit            | List\[float\]                    | -                                          | The parameter upper limit x_u.                                                                                                             |
| eq_func                    | Callable\[List\[float\]\]        | None                                       | The equality constraint function h(x).                                                                                                     |
| eq_values                  | List\[float\]                    | None                                       | The equality constraint values e_x.                                                                                                        |
| ineq_func                  | Callable\[List\[float\]\]        | None                                       | The inequality constraint function g(x).                                                                                                   |
| ineq_lower_bounds          | List\[float\]                    | None                                       | The inequality constraint lower limit g_l.                                                                                                 |
| ineq_upper_bounds          | List\[float\]                    | None                                       | The inequality constraint upper limit g_l.                                                                                                 |
| number_of_restarts         | int                              | 1                                          | The `number_of_restarts` best evaluation results are used to run pysolnp `number_of_restarts` times.                                       |
| number_of_simulations      | int                              | 20000                                      | Sets how many randomly generated starting guesses we generate and evaluate with the evaluation function.                                   |
| number_of_processes        | int                              | None                                       | Sets how many parallel processes to run when solving the problem. If None the problem is solved in the main processes.                     |
| start_guess_sampling       | List\[Distribution\] or Sampling | None                                       | A list of distributions for generating starting values, one distribution for each parameter. If None, the Uniform distribution is used.*** |
| seed                       | int                              | None                                       | By default the MT19937 Generator is used with timestamp-seed. Optionally an integer seed can be supplied.                                  |
| evaluation_type            | EvaluationType or int            | EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ | Selects the evaluation type from the pygosolnp.EvaluationType enum.                                                                        |
| pysolnp_rho                | float                            | 1.0                                        | pysolnp parameter: Penalty weighting scalar for infeasability in the augmented objective function.**                                       |
| pysolnp_max_major_iter     | int                              | 400                                        | pysolnp parameter: Maximum number of outer iterations.                                                                                     |
| pysolnp_max_minor_iter     | int                              | 800                                        | pysolnp parameter: Maximum number of inner iterations.                                                                                     |
| pysolnp_delta              | float                            | 1e-07                                      | pysolnp parameter: Step-size for forward differentiation.                                                                                  |
| pysolnp_tolerance          | float                            | 1e-08                                      | pysolnp parameter: Relative tolerance on optimality.                                                                                       |
| debug                      | bool                             | False                                      | If set to true some debug output will be printed.                                                                                          |

*Defaults for configuration parameters are based on the defaults for Rsolnp.<br>

**Higher values means the solution will bring the solution into the feasible region with higher weight. Very high values might lead to numerical ill conditioning or slow down convergence.

***Supply an instance of a class that inherits the abstract class `pygosolnp.sampling.Sampling` to provide starting guesses, see below for examples:
1) [/python_examples/example_grid_sampling.py](https://github.com/KristerSJakobsson/pygosolnp/blob/main/python_examples/example_grid_sampling.py) - Uses Scikit-optimize to generate grid-style random starting guesses.
2) [/python_examples/example_truncated_normal.py](https://github.com/KristerSJakobsson/pygosolnp/blob/main/python_examples/example_truncated_normal.py) - Uses Scipy random to generate Truncated Normal random numbers using the PCG64 generator. 

Output:
The function returns the `pygosolnp.Results` with the below properties.

| Property           | Type               | Description                                           |
| -------------------|:-------------------|-------------------------------------------------------|
| best_solution      | Optional\[Result\] | The best local optimum found for the problem.         |
| all_results        | List\[Result\]     | All restarts and their corresponding local optimum.   |
| starting_guesses   | List\[float\]      | All the randomized starting parameters.               |

Each named tuple `pygosolnp.Result` has the below properties.

| Property           | Type           | Description                                                 |
| -------------------|:---------------|-------------------------------------------------------------|
| obj_value          | float          | The value of the objective function at local optimum f(x*). |
| parameters         | List\[float\]  | A list of parameters for the local optimum x*.              |
| converged          | bool           | Boolean which indicates if the solution is within bounds.   |

## Multiprocessing
pygosolnp supports multi-processing (not multi-threading!) using the standard multi-processing library.
This is an advanced feature, please read up on this before using it!

There are various things to consider in order to get time- and memory-efficient executions:
- Multiprocessing will spawn processes, this consumes time and memory, if your problem is small then run it single-threaded! 
- Your operating system, notably Linux works better with multiprocessing than Windows.
- All function must be picklable (for example global functions, local lambdas will not work)

## Authors

* **Krister S Jakobsson** - *Implementation* - krister.s.jakobsson@gmail.com

## License

This project is licensed under the Boost License - see the [license](https://github.com/KristerSJakobsson/pygosolnp/blob/main/LICENSE) file for details.

## Acknowledgments

* **Yinyu Ye** -  Publisher and mastermind behind the original SOLNP algorithm,
[Original Sources](https://web.stanford.edu/~yyye/matlab/)
* **Alexios Ghalanos and Stefan Theussl** - The people behind RSOLNP and GOSOLNP,
