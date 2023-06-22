"""
Sample 2D test functions taken from 
https://en.wikipedia.org/wiki/Test_functions_for_optimization.
These allow to nicely plot and investigate optimization algorithms.
"""

import numpy as np

# abstract base class for implementing interface
from abc import ABC, abstractmethod
from . import utils


class Testfunction2D(ABC):
    """
    Abstract base class (ABC) that implements the interface of any 2D test
    function. Most importantly, this enforces each function to have a name and
    the area of the search space.
    """

    def __init__(self, name, area, opt_pos, opt_val, goal='min'):
        self.name = name
        self.area = utils.validate_area(area)
        self.opt_pos = opt_pos
        self.opt_val = opt_val
        self.goal = utils.validate_goal(goal)

    @abstractmethod
    def objective(self, x, y):
        pass

class Ackley(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Ackley',
            area=np.array([[-5., 5.], [-5., 5.]]),
            opt_pos=np.array([0., 0.]),
            opt_val=0.0
        )

    def objective(self, x, y):
        out = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        out -= np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        out += np.e + 20
        return out

class Sphere(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Sphere',
            area=np.array([[-50.0, 50.0], [-50.0, 50.0]]),
            opt_pos=np.array([0., 0.]),
            opt_val=0.0
        )

    def objective(self, x, y):
        return x**2 + y**2

class Beale(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Beale',
            area=np.array([[-4.5, 4.5], [-4.5, 4.5]]),
            opt_pos=np.array([3., 0.5]),
            opt_val=0.0
        )

    def objective(self, x, y):
        out = (1.5 - x + x * y) ** 2
        out += (2.25 - x + x * y ** 2) ** 2
        out += (2.625 - x + x * y ** 3) ** 2
        return out

class Himmelblau(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Himmelblau',
            area=np.array([[-5., 5.], [-5., 5.]]),
            opt_pos=np.array([[3., 2.], [-2.805118, 3.131312], [-3.779310, -3.283186], [3.584428, -1.848126]]),
            opt_val=0.0
        )

    def objective(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
