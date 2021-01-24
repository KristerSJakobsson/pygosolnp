# ----------------------------------------------------------------------------------
# Some Problems in Global Optimization
# ----------------------------------------------------------------------------------


# The Permutation Function has unique solution f(x) = 0 when x_i = i
def permutation_function(data, n=4, b=0.5):
    result1 = 0
    for index1 in range(1, n + 1):
        result2 = 0
        for index2 in range(1, n + 1):
            result2 += ((pow(index2, index1) + b) * (pow(data[index2 - 1] / index2, index1) - 1))
        result1 += pow(result2, 2)
    return result1


parameter_lower_bounds = [-4.0] * 4
parameter_upper_bounds = [4.0] * 4
