from math import sqrt


# ----------------------------------------------------------------------------------
# Some Problems in Global Optimization
# ----------------------------------------------------------------------------------

# Distribution of Electrons on a Sphere
# See https://www.mcs.anl.gov/~more/cops/cops3.pdf

def obj_func(data):
    electron = Electron(number_of_charges=25)
    return electron.objective_function(data=data)


def eq_func(data):
    electron = Electron(number_of_charges=25)
    return electron.equality_function(data=data)


class Electron:
    def __init__(self, number_of_charges):
        self.__number_of_charges = number_of_charges

    @property
    def number_of_charges(self):
        return self.__number_of_charges

    @property
    def number_of_parameters(self):
        return self.__number_of_charges * 3

    @property
    def parameter_lower_bound(self):
        return [-1] * self.number_of_parameters

    @property
    def parameter_upper_bound(self):
        return [1] * self.number_of_parameters

    def objective_function(self, data):
        n = self.number_of_charges
        x = data[0:n]
        y = data[n:2 * n]
        z = data[2 * n:3 * n]

        result = 0.0
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                result += 1.0 / sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2)

        return result

    def equality_function(self, data):
        n = self.number_of_charges
        x = data[0:n]
        y = data[n:2 * n]
        z = data[2 * n:3 * n]
        result = [None] * n
        for i in range(0, n):
            result[i] = x[i] ** 2 + y[i] ** 2 + z[i] ** 2

        return result

    @property
    def equality_constraint_bounds(self):
        return [1] * self.number_of_charges
