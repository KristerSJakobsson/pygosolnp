from enum import Enum
from multiprocessing import Array
from typing import Callable, List, Optional, Union, Any

from src.sampling import Distribution


class EvaluationType(Enum):
    OBJECTIVE_FUNC_EXCLUDE_INEQ = 1  # Exclude any guesses that violate the objective function and the evaluate with objective function
    PENALTY_BARRIER_FUNCTION = 2  # Use the penalty barrier function to evaluate objective function


class ProblemModel:

    def __init__(self,
                 obj_func: Callable,
                 par_lower_limit: List[float],
                 par_upper_limit: List[float],
                 number_of_restarts: int,
                 number_of_simulations: int,
                 eq_func: Optional[Callable] = None,
                 eq_values: Optional[List[float]] = None,
                 ineq_func: Optional[Callable] = None,
                 ineq_lower_bounds: Optional[List[float]] = None,
                 ineq_upper_bounds: Optional[List[float]] = None,
                 rho: float = 1.0,
                 max_major_iter: int = 10,
                 max_minor_iter: int = 10,
                 delta: float = 1e-05,
                 tolerance: float = 0.0001,
                 debug: bool = False,
                 number_of_processes: Optional[int] = 5,
                 random_number_distribution: Optional[List[Distribution]] = None,
                 random_number_seed: Optional[int] = None,
                 evaluation_type: EvaluationType = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ):
        self.__obj_func = obj_func
        self.__par_lower_limit = par_lower_limit
        self.__par_upper_limit = par_upper_limit
        self.__number_of_restarts = number_of_restarts
        self.__number_of_simulations = number_of_simulations
        self.__eq_func = eq_func
        self.__eq_values = eq_values
        self.__ineq_func = ineq_func
        self.__ineq_lower_bounds = ineq_lower_bounds
        self.__ineq_upper_bounds = ineq_upper_bounds
        self.__rho = rho
        self.__max_major_iter = max_major_iter
        self.__max_minor_iter = max_minor_iter
        self.__delta = delta
        self.__tolerance = tolerance
        self.__debug = debug
        self.__number_of_processes = number_of_processes
        self.__random_number_distribution = random_number_distribution
        self.__random_number_seed = random_number_seed
        self.__evaluation_type = evaluation_type

    @property
    def obj_func(self):
        return self.__obj_func

    @property
    def number_of_restarts(self) -> int:
        return self.__number_of_restarts

    @property
    def number_of_simulations(self) -> int:
        return self.__number_of_simulations

    @property
    def number_of_samples(self) -> int:
        return self.__number_of_restarts * self.__number_of_simulations

    @property
    def ineq_func(self) -> Callable:
        return self.__ineq_func

    @property
    def par_lower_limit(self) -> Union[List, Array]:
        return self.__par_lower_limit

    @property
    def par_upper_limit(self) -> Union[List, Array]:
        return self.__par_upper_limit

    @property
    def eq_values(self) -> Union[List, Array]:
        return self.__eq_values

    @property
    def ineq_lower_bounds(self) -> Union[List, Array]:
        return self.__ineq_lower_bounds

    @property
    def ineq_upper_bounds(self) -> Union[List, Array]:
        return self.__ineq_upper_bounds

    @property
    def delta(self) -> float:
        return self.__delta

    @property
    def rho(self) -> float:
        return self.__rho

    @property
    def tolerance(self) -> float:
        return self.__tolerance

    @property
    def max_major_iter(self) -> int:
        return self.__max_major_iter

    @property
    def max_minor_iter(self) -> int:
        return self.__max_minor_iter

    @property
    def debug(self) -> bool:
        return self.__debug

    @property
    def evaluation_type(self) -> EvaluationType:
        return self.__evaluation_type

    @property
    def number_of_simulations(self) -> int:
        return self.__number_of_simulations

    @property
    def number_of_parameters(self) -> int:
        return len(self.__par_lower_limit)

    def validate(self):
        if type(self.__par_lower_limit) is not list or type(self.__par_upper_limit) is not list:
            raise ValueError("Parameter lower and upper bounds are required for gosolnp")

        if len(self.__par_lower_limit) != len(self.__par_upper_limit):
            raise ValueError("The length of lower and upper bounds are not the same")

        if type(self.__random_number_distribution) is list and len(self.__random_number_distribution) != len(
                self.__par_lower_limit):
            raise ValueError(
                "random_number_distribution input must be either one distribution or an array of distributions.")

    def check_solution_feasibility(self, par_found_solution):
        if any(value < self.__par_lower_limit[index] - self.__tolerance or self.__par_upper_limit[
            index] + self.__tolerance < value for index, value in
               enumerate(par_found_solution)):
            return False

        if self.__eq_func is not None:
            equality_function_values = self.__eq_func(par_found_solution)
            if any(value < self.__eq_values[index] - self.__tolerance or self.__eq_values[
                index] + self.__tolerance < value for index, value in
                   enumerate(equality_function_values)):
                return False

        if self.__ineq_func is not None:
            inequality_function_values = self.__ineq_func(par_found_solution)
            if any(value < self.__ineq_lower_bounds[index] - self.__tolerance or self.__ineq_upper_bounds[
                index] + self.__tolerance < value for
                   index, value in enumerate(inequality_function_values)):
                return False

        return True

