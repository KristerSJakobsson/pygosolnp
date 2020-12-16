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
    number_of_restarts=2,
    number_of_simulations=200,
    number_of_processes=None,
    seed_or_generator=443,
    pysolnp_max_major_iter=100,
    evaluation_type=pygosolnp.EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
    debug=False)

all_results = results.all_results
print("; ".join([f"Solution {index + 1}: {solution.obj_value}" for index, solution in enumerate(all_results)]))
best_solution = results.best_solution
print(f"Best solution {best_solution.obj_value} for parameters {best_solution.parameters}.")
```

Output:
```
Solution 1: 244.5323981409454; Solution 2: 243.90905145098753
Best solution 243.90905145098753 for parameters [0.9842020713431833, 0.28279657501813066, -0.6550309607672415, -0.35956474685388357, -0.48262010671846156, -0.06173919085442064, 0.6312458462439161, -0.02618476879250702, 0.7504655316065533, -0.7841259302019284, -0.01803010989498263, -0.5580446479715245, -0.044635961596336565, 0.3877663346320099, 0.7238518934217706, 0.8779123988423063, -0.7410898834735291, -0.24565333576960174, 0.37567324416613146, -0.3604479630690014, -0.9805252621984456, 0.6312695698146041, -0.8519258909782209, 0.3653362742400498, 0.0911961003178957, 0.15723506918554647, 0.794412194065582, 0.6892215297759574, -0.8849672831699511, 0.7657816333627421, 0.3400596536523402, 0.3375867688085594, -0.5835864218147403, 0.2781718344447138, -0.3722891612035071, 0.9953279138691365, 0.153694469699745, 0.6789381318882565, -0.772058966894386, -0.3953314276418762, -0.42511946411852836, 0.1527487725756057, -0.8845591933765159, -0.8995101024652117, -0.3881497709781692, 0.19559533702225373, 0.7662458064054968, -0.4745210574540679, -0.21501758603853763, -0.007916738485806953, 0.08138442013871594, -0.5375270922553241, 0.30969049839376706, 0.2958821670249552, -0.42503688739335593, -0.9383750406680831, 0.6982577353344747, 0.8116288640651308, -0.5995180881742376, 0.4965353054418313, 0.09485407004856716, 0.8154533653415039, 0.7328373956527733, -0.5035496772750786, 0.5654657634442658, -0.22032555398627657, -0.653799361064562, -0.3964965955481358, 0.22304975562065002, -0.8481845475372783, 0.01768261465276446, 0.11985870061630192, -0.2214772810065067, -0.9057024457948485, 0.9958015110499229].
```

## Parameters
The basic signature is:
```python
pygosolnp.solve(obj_func: Callable,
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
                random_number_distribution: Optional[List[Distribution]] = None,
                seed_or_generator: Union[None, int, Generator] = None,
                evaluation_type: Union[EvaluationType, int] = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ,
                pysolnp_rho: float = 1.0,
                pysolnp_max_major_iter: int = 10,
                pysolnp_max_minor_iter: int = 10,
                pysolnp_delta: float = 1e-05,
                pysolnp_tolerance: float = 0.0001,
                debug: bool = False) -> Results
```

Inputs:

| Parameter                  | Type                          | Default value*                             | Description                                                                                                                             |
| ---------------------------|:------------------------------|:-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| obj_func                   | Callable\[List, float\]       | -                                          | The objective function f(x) to minimize.                                                                                                |
| par_lower_limit            | List                          | -                                          | The parameter lower limit x_l.                                                                                                          |
| par_upper_limit            | List                          | -                                          | The parameter upper limit x_u.                                                                                                          |
| eq_func                    | Callable\[List, float\]       | None                                       | The equality constraint function h(x).                                                                                                  |
| eq_values                  | List                          | None                                       | The equality constraint values e_x.                                                                                                     |
| ineq_func                  | Callable\[List, float\]       | None                                       | The inequality constraint function g(x).                                                                                                |
| ineq_lower_bounds          | List                          | None                                       | The inequality constraint lower limit g_l.                                                                                              |
| ineq_upper_bounds          | List                          | None                                       | The inequality constraint upper limit g_l.                                                                                              |
| number_of_restarts         | int                           | 1                                          | The `number_of_restarts` best evaluation results are used to run pysolnp `number_of_restarts` times.                                    |
| number_of_simulations      | int                           | 20000                                      | Sets how many randomly generated starting guesses we generate and evaluate with the evaluation function.                                |
| number_of_processes        | int                           | None                                       | Sets how many parallel processes to run when solving the problem. If None the problem is solved in the main processes.                  |
| random_number_distribution | List\[Distribution\]          | None                                       | A list of distributions for generating starting values, one distribution for each parameter. If None, the Uniform distribution is used. |
| seed_or_generator          | Union\[None, int, Generator\] | None                                       | By default the PCG64 Generator is used with timestamp-seed. Optionally an integer seed or a numpy.Generator can be supplied.            |
| evaluation_type            | Union\[int, EvaluationType\]  | EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ | Selects the evaluation type from the pygosolnp.EvaluationType enum.                                                                     |
| pysolnp_rho                | float                         | 1.0                                        | pysolnp parameter: Penalty weighting scalar for infeasability in the augmented objective function.**                                    |
| pysolnp_max_major_iter     | int                           | 400                                        | pysolnp parameter: Maximum number of outer iterations.                                                                                  |
| pysolnp_max_minor_iter     | int                           | 800                                        | pysolnp parameter: Maximum number of inner iterations.                                                                                  |
| pysolnp_delta              | float                         | 1e-07                                      | pysolnp parameter: Step-size for forward differentiation.                                                                               |
| pysolnp_tolerance          | float                         | 1e-08                                      | pysolnp parameter: Relative tolerance on optimality.                                                                                    |
| debug                      | bool                          | False                                      | If set to true some debug output will be printed.                                                                    |

*Defaults for configuration parameters are based on the defaults for Rsolnp.<br>
**Higher values means the solution will bring the solution into the feasible region with higher weight. Very high values might lead to numerical ill conditioning or slow down convergence.

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
This is an advanced feature, please read up on this before using it.

There are various things to consider in order to get time- and memory-efficient executions:
- Multiprocessing will spawn processes, this consumes time and memory, if your problem is small then run it single-threaded! 
- Your operating system, notably Linux works better with multiprocessing than Windows
- All function must be picklable (for example global functions, local lambdas will not work)

## Authors

* **Krister S Jakobsson** - *Implementation* - krister.s.jakobsson@gmail.com

## License

This project is licensed under the Boost License - see the [license](LICENSE.md) file for details.

## Acknowledgments

* **Yinyu Ye** -  Publisher and mastermind behind the original SOLNP algorithm,
[Original Sources](https://web.stanford.edu/~yyye/matlab/)
* **Alexios Ghalanos and Stefan Theussl** - The people behind RSOLNP and GOSOLNP,
