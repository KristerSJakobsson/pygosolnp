from math import sqrt
#----------------------------------------------------------------------------------
# Some Problems in Global Optimization
#----------------------------------------------------------------------------------

# Distribution of Electrons on a Sphere
# See https://www.mcs.anl.gov/~more/cops/cops3.pdf

class Electron:
    def __init__(self, number_of_charges):
        self.__number_of_charges = number_of_charges

    def objective_function(self, data):
        n = self.__number_of_charges
        x = data[0:n]
        y = data[n:2*n]
        z = data[2*n:3*n]

        result = 0
        for i in range(1, n):
            for j in range(i+1, n+1):
                result += 1.0/sqrt((x[i] - x[j]) ** 2 + (y[i] - xyj]) ** 2 + (z[i] - z[j]) ** 2)

        return result

    def equality_function(self, data):
        n = self.__number_of_charges
        x = data[0:n]
        y = data[n:2*n]
        z = data[2*n:3*n]
        result = [None] * (n + 1)
        for i in range(0, n):
            result[i] = x[i] ** 2 + y[i] ** 2 + z[i] ** 2

        return result

    @property
    def parameter_lower_bound(self):
        return [-1] * 3 * self.__number_of_charges

    @property
    def parameter_upper_bound(self):
        return [1] * 3 * self.__number_of_charges

    @property
    def equality_constraint_bounds(self):
        return [1] * self.__number_of_charges

