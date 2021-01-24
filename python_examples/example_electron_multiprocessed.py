import pygosolnp
from math import sqrt
import time

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

if __name__ == '__main__':
    start = time.time()
    results = pygosolnp.solve(obj_func=obj_func,
                              eq_func=eq_func,
                              eq_values=equality_constraints,
                              par_lower_limit=parameter_lower_bounds,
                              par_upper_limit=parameter_upper_bounds,
                              number_of_restarts=20,
                              number_of_simulations=20000,
                              number_of_processes=4,  # Simulations and processes will be executed in 4 processes
                              seed=443,
                              pysolnp_max_major_iter=100,
                              debug=False)

    end = time.time()

    all_results = results.all_results
    print("; ".join([f"Solution {index + 1}: {solution.obj_value}" for index, solution in enumerate(all_results)]))
    best_solution = results.best_solution
    print(f"Best solution {best_solution.obj_value} for parameters {best_solution.parameters}.")
    print(f"Elapsed time: {end - start} s")

# Output below:
# Solution 1: 244.1550118432253; Solution 2: 243.9490050190484; Solution 3: 185.78533081425041; Solution 4: 244.07921194485854; Solution 5: 216.19236253370485; Solution 6: 194.1742137471891; Solution 7: 258.6157748268509; Solution 8: 205.72538678938517; Solution 9: 244.0944480181356; Solution 10: 217.4090464122706; Solution 11: 201.58045387715478; Solution 12: 247.70691375326325; Solution 13: 243.92615570955812; Solution 14: 192.3944392661305; Solution 15: 243.93657263760585; Solution 16: 247.17924771908508; Solution 17: 244.06529702108125; Solution 18: 244.29427536763717; Solution 19: 199.69130383979302; Solution 20: 243.99315264179037
# Best solution 243.92615570955812 for parameters [0.8726149386907173, 0.1488320711741995, -0.8215181712229778, 0.8597822831494584, -0.265961670940264, -0.6664127144955102, -0.6029702658967409, 0.2867960203292267, -0.04380531711098636, 0.9519854892760677, -0.39592769694574026, -0.2160514547351913, -0.21416235954836016, 0.4338472533837847, -0.9411378567701716, 0.6418976636970082, 0.014864034847848012, 0.6981416769347426, 0.4413252856284809, -0.5267725521555819, -0.9148568048943023, -0.5831731928212042, 0.47570915153781534, 0.4089885176760918, 0.008471540399374077, -0.36287443863890595, 0.8618964461129363, 0.5476494687199884, -0.3309316231117961, 0.9582851670742292, -0.6505818085537286, 0.2793946112676732, -0.7596998666078645, 0.65142774983249, 0.30572406841664945, -0.1736400992779951, -0.2357569641249718, -0.9762296783338298, 0.8894784482368485, -0.21768032982807542, 0.44966067028074935, 0.359898210796523, 0.3932146838134686, -0.25429503229562933, -0.6621520897149067, 0.0002565729867240561, 0.6081775900274631, -0.8731755460834034, -0.07630776960802095, -0.7462707639808169, 0.32690759610807246, 0.4847543563757037, -0.15870866693945487, -0.38892531575475037, -0.10466177783304143, 0.36421374544164403, -0.7472412325499505, -0.583622807257543, -0.7574487346380878, -0.01614470971763483, 0.9017203154504035, -0.9474931851459008, -0.03334319523220503, -0.14354857449259437, -0.258603947854119, -0.6211074642796408, 0.9328743112042068, 0.5983190378042788, 0.860564215444357, -0.5329857672153024, 0.403783074281117, 0.538582127861995, 0.1061505899839121, -0.9093445419255864, 0.6656150775217203].
# Elapsed time: 596.5835165977478 ms
