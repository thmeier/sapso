"""
Sample 2D test functions taken from 
https://en.wikipedia.org/wiki/Test_functions_for_optimization.
These allow to nicely plot and investigate optimization algorithms.
"""

import numpy as np

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
        self.opt_val = opt_val
        self.opt_pos = opt_pos
        self.goal = utils.validate_goal(goal)

    @abstractmethod
    def objective(self, x, y):
        pass

class Ackley(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Ackley',
            area=np.array([[-5., 5.], [-5., 5.]]),
            opt_val=0.0,
            opt_pos=np.array([
                (0., 0.)
            ]),
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
            opt_val=0.0,
            opt_pos=np.array([
                (0., 0.)
            ]),
        )

    def objective(self, x, y):
        return x**2 + y**2

class Beale(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Beale',
            area=np.array([[-4.5, 4.5], [-4.5, 4.5]]),
            opt_val=0.0,
            opt_pos=np.array([
                (3., 0.5)
            ]),
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
            opt_val=0.0,
            opt_pos=np.array([
                (3., 2.),
                (-2.805118, 3.131312),
                (-3.779310, -3.283186),
                (3.584428, -1.848126)
            ]),
        )

    def objective(self, x, y):
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

class Eggholder(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Eggholder',
            area=np.array([[-512., 512.], [-512., 512.]]),
            opt_val=-959.6407,
            opt_pos=np.array([
                (512., 404.2319)
            ]),
        )

    def objective(self, x, y):
        y += 47
        out = -y * np.sin(np.sqrt(np.abs(x / 2 + y))) 
        out -= x * np.sin(np.sqrt(np.abs(x - y)))
        return out

class CrossInTray(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Cross-in-tray',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=-2.06261,
            opt_pos=np.array([
                (1.34941, 1.34941),
                (1.34941, -1.34941),
                (-1.34941, 1.34941),
                (-1.34941, -1.34941),
            ]),
        )

    def objective(self, x, y):
        out = np.exp(np.abs(100 - np.sqrt(x**2 + y**2) / np.pi))
        out = np.abs(np.sin(x) * np.sin(y) * out) + 1
        out = -0.0001 * np.power(out, 0.1)
        return out

class SchaffnerNo2(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Schaffner-No2',
            area=np.array([[-100., 100.], [-100., 100.]]),
            opt_val=0.,
            opt_pos=np.array([
                (0., 0.)
            ])
        )

    def objective(self, x, y):
        out = np.sin(x**2 - y**2)**2 - 0.5
        out = out / (1 + 0.001 * (x**2 + y**2))**2 + 0.5
        return out

class SchaffnerNo4(Testfunction2D):
    def __init__(self):
        super().__init__(
            name='Schaffner-No4',
            area=np.array([[-100., 100.], [-100., 100.]]),
            opt_val=0.292579,
            opt_pos=np.array([
                (0., 1.25313)
                (0., -1.25313)
                (1.25313, 0.)
                (-1.25313, 0.)
            ])
        )

    def objective(self, x, y):
        out = np.cos(np.sin(np.abs(x**2 - y**2)))**2 - 0.5
        out = out / (1 + 0.001 * (x**2 + y**2))**2 + 0.5
        return out
