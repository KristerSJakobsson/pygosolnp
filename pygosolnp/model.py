from enum import Enum
from multiprocessing import Array
from typing import Callable, Optional, Union, List

from pygosolnp.sampling import Distribution, DefaultSampling


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
                 start_guess_sampling: Union[None, List[Distribution], DefaultSampling] = None,
                 evaluation_type: Union[EvaluationType, int] = EvaluationType.OBJECTIVE_FUNC_EXCLUDE_INEQ):
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
        self.__start_guess_sampling = start_guess_sampling
        self.__evaluation_type = EvaluationType(evaluation_type)

    @property
    def obj_func(self):
        return self.__obj_func

    @property
    def number_of_restarts(self) -> int:
        return self.__number_of_restarts

    @property
    def number_of_evaluations(self) -> int:
        return self.__number_of_simulations

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
    def sample_size(self) -> int:
        return len(self.__par_lower_limit)

    @property
    def eq_func(self) -> Callable:
        return self.__eq_func

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
    def number_of_parameters(self) -> int:
        return len(self.__par_lower_limit)

    @property
    def has_eq_bounds(self) -> bool:
        return self.__eq_func is not None and \
               self.__eq_values is not None

    @property
    def has_ineq_bounds(self) -> bool:
        return self.__ineq_func is not None and \
               self.__ineq_lower_bounds is not None and \
               self.__ineq_upper_bounds is not None

    @property
    def start_guess_sampling(self):
        return self.__start_guess_sampling

    def validate(self):
        mandatory_data = [self.__obj_func, self.__par_lower_limit, self.__par_upper_limit]
        if any(data is None for data in mandatory_data):
            raise ValueError(
                "obj_func, par_lower_limit and par_upper_limit are required for pygosolnp to function")

        if not callable(self.__obj_func):
            raise ValueError("obj_func must be callable")

        if not hasattr(self.__par_lower_limit, '__len__'):
            raise ValueError("par_lower_limit does not have a fixed length, is it an list or array?")

        if not hasattr(self.__par_upper_limit, '__len__'):
            raise ValueError("par_upper_limit does not have a fixed length, is it an list or array?")

        if len(self.__par_lower_limit) != len(self.__par_upper_limit):
            raise ValueError("par_lower_limit and par_upper_limit bounds are not of the same length")

        if hasattr(self.__start_guess_sampling, '__len__') and \
                len(self.__start_guess_sampling) != len(self.__par_lower_limit):
            raise ValueError(
                "random_number_distribution input must be either None, an array of distributions with the same length as par_lower_limit and par_upper_limit or a Sampling instance")

        eq_data = [self.__eq_func, self.__eq_values]
        if not (all(data is None for data in eq_data) or all(data is not None for data in eq_data)):
            raise ValueError(
                "For equality constrained problems, please provide both eq_func and eq_values, or alternatively set both to None if not applicable")

        if self.__eq_func is not None and not callable(self.__eq_func):
            raise ValueError("For equality constrained problems, please supply a callable eq_func")

        if self.__eq_values is not None and not hasattr(self.__eq_values, '__len__'):
            raise ValueError("For equality constrained problems, please supply an fixed-length eq_values")

        ineq_data = [self.__ineq_func, self.__ineq_lower_bounds, self.__ineq_upper_bounds]
        if not (all(data is None for data in ineq_data) or all(data is not None for data in ineq_data)):
            raise ValueError(
                "For inequality constrained problems, please provide ineq_func, ineq_lower_bound and ineq_upper_bounds, or alternatively set all to None if not applicable")

        if self.__ineq_func is not None and not callable(self.__ineq_func):
            raise ValueError("For inequality constrained problems, please supply a callable ineq_func")

        if self.__ineq_lower_bounds is not None and not hasattr(self.__ineq_lower_bounds, '__len__'):
            raise ValueError("For inequality constrained problems, please supply an fixed-length ineq_lower_bounds")

        if self.__ineq_upper_bounds is not None and not hasattr(self.__ineq_upper_bounds, '__len__'):
            raise ValueError("For inequality constrained problems, please supply an fixed-length ineq_upper_bounds")

        if (self.__ineq_lower_bounds is not None) and (self.__ineq_upper_bounds is not None) and (
                len(self.__ineq_lower_bounds) != len(self.__ineq_upper_bounds)):
            raise ValueError(
                "For inequality constrained problems, please make sure that ineq_lower_bound is of the same length as ineq_upper_bound")

        if self.__number_of_simulations < 1:
            raise ValueError("number_of_simulations needs to be a positive integer value")

        if self.__number_of_restarts < 1:
            raise ValueError("number_of_restarts needs to be a positive integer value")

        if self.__number_of_processes is not None and self.__number_of_processes < 1:
            raise ValueError(
                "number_of_processes needs to be a positive integer value and is recommended to be greater than or equal to 2")

        if type(self.__rho) is not float:
            raise ValueError("pysolnp_rho needs to be a float value")

        if type(self.__max_major_iter) is not int:
            raise ValueError("pysolnp_max_major_iter needs to be a int value")

        if self.__max_major_iter <= 0:
            raise ValueError("pysolnp_max_major_iter needs to be a positive value")

        if type(self.__max_minor_iter) is not int:
            raise ValueError("pysolnp_max_minor_iter needs to be a int value")

        if self.__max_minor_iter <= 0:
            raise ValueError("pysolnp_max_minor_iter needs to be a positive value")

        if type(self.__delta) is not float:
            raise ValueError("pysolnp_delta needs to be a float value")

        if type(self.__tolerance) is not float:
            raise ValueError("pysolnp_tolerance needs to be a float value")

        if type(self.__debug) is not bool:
            raise ValueError("debug needs to be a boolean value")

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
