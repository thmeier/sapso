# all functions taken from
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

import numpy as np

"""
minimum : f(0, 0) = 0
range   : [-5, 5]
"""
def ackley(x, y):
    out = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
    out -= np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    out += np.e + 20
    return out

"""
minimum : f(0, 0) = 0
range   : [-inf, inf]
"""
def sphere(x, y):
    return x**2 + y**2

"""
minimum : f(3, 0.5) = 0
range   : [-4.5, 4.5]
"""
def beale(x, y):
    return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

"""
minimum : f(0, 0) = 0
range   : [-inf, inf]
"""
def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
