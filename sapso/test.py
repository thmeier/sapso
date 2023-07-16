"""
Sample 2D test functions taken from 
https://en.wikipedia.org/wiki/Test_functions_for_optimization.
These allow to nicely plot and investigate optimization algorithms.
"""

import numpy as np

from abc import ABC, abstractmethod

from . import utils

__all__ = [
    'Testfunction2D', 'Ackley', 'Sphere', 'Beale', 'Himmelblau', 
    'Eggholder', 'CrossInTray', 'SchaffnerNo2', 'SchaffnerNo4'
]

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
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
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

class Ackley_safe(Testfunction2D):
    """
    Same as Ackley, but subtracted 10 to have optimum at -10
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Ackley',
            area=np.array([[-5., 5.], [-5., 5.]]),
            opt_val=-10.0,
            opt_pos=np.array([
                (0., 0.)
            ]),
        )

    def objective(self, x, y):
        out = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        out -= np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        out += np.e + 20
        out -= 10.0
        return out

class Sphere(Testfunction2D):
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
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
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
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
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
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
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
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
        out = y + 47
        out = -out * np.sin(np.sqrt(np.abs(x / 2 + out))) 
        out -= x * np.sin(np.sqrt(np.abs(x - y)))
        return out

class CrossInTray(Testfunction2D):
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
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
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Schaffner No2',
            #area=np.array([[-50., 50.], [-50., 50.]]),
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

class SchaffnerNo2_safe(Testfunction2D):
    """
    Same as SchaffnerNo2, but subtracted 1 to have optimum at -1
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Schaffner No2',
            #area=np.array([[-50., 50.], [-50., 50.]]),
            area=np.array([[-100., 100.], [-100., 100.]]),
            opt_val=-1.,
            opt_pos=np.array([
                (0., 0.)
            ])
        )

    def objective(self, x, y):
        out = np.sin(x**2 - y**2)**2 - 0.5
        out = out / (1 + 0.001 * (x**2 + y**2))**2 + 0.5
        out -= 1.0
        return out

class SchaffnerNo4(Testfunction2D):
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Schaffner No4',
            area=np.array([[-100., 100.], [-100., 100.]]),
            opt_val=0.292579,
            opt_pos=np.array([
                (0., 1.25313),
                (0., -1.25313),
                (1.25313, 0.),
                (-1.25313, 0.),
            ])
        )

    def objective(self, x, y):
        out = np.cos(np.sin(np.abs(x**2 - y**2)))**2 - 0.5
        out = out / (1 + 0.001 * (x**2 + y**2))**2 + 0.5
        return out

class BukinNo6(Testfunction2D):
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Bukin No6',
            area=np.array([[-15., -5.], [-3., 3.]]),
            opt_val=0.0,
            opt_pos=np.array([
                (-10.0, 1.),
            ])
        )

    def objective(self, x, y):
        out = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
        return out

class BukinNo6_safe(Testfunction2D):
    """
    Same as BukinNo6, but subtracted 10 to have optimum at -10
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Bukin No6',
            area=np.array([[-15., -5.], [-3., 3.]]),
            opt_val=-10.0,
            opt_pos=np.array([
                (-10.0, 1.),
            ])
        )

    def objective(self, x, y):
        out = 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
        out -= 10.0
        return out

class HoelderTable(Testfunction2D):
    """
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    def __init__(self):
        super().__init__(
            name='Hoelder Table',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=-19.2085,
            opt_pos=np.array([
                (8.05502, 9.66459),
                (-8.05502, 9.66459),
                (8.05502, -9.66459),
                (-8.05502, -9.66459),
            ])
        )

    def objective(self, x, y):
        out = np.exp(np.abs(1.0 - np.sqrt(x**2 + y**2) / np.pi))
        out = -np.abs(np.sin(x) * np.cos(y) * out)
        return out

class Michalewicz(Testfunction2D):
    """
    https://www.sfu.ca/~ssurjano/michal.html
    """
    def __init__(self):
        super().__init__(
            name='Michalewicz',
            area=np.array([[0., 3.], [0., 3.]]),
            opt_val=-1.8013,
            opt_pos=np.array([
                (2.20, 1.57),
            ])
        )

    def objective(self, x, y):
        m = 10
        out = np.sin(x) * np.sin(x**2 / np.pi)**(2 * m)
        out += np.sin(y) * np.sin(2 * y**2 / np.pi)**(2 * m)
        out *= -1
        return out

class Layeb04(Testfunction2D):
    """
    aka Crossfly, taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf
    """
    def __init__(self):
        super().__init__(
            name='Layeb04',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=np.log(0.001)-1.,
            # opt_pos can be reproduced by
            # np.unique(np.concatenate(
            #   [ [(u,v), (v,u)] for u in us for v in vs ]
            # ), axis=0)
            # where us, vs = np.zeros(4), [(2*k-1)*np.pi for k in range(-1,3)]
            opt_pos=np.array([
                ( 0.        , -9.42477796),
                ( 0.        , -3.14159265),
                ( 0.        ,  3.14159265),
                ( 0.        ,  9.42477796),
                (-9.42477796,  0.        ),
                (-3.14159265,  0.        ),
                ( 3.14159265,  0.        ),
                ( 9.42477796,  0.        ),
            ])
        )

    def objective(self, x, y):
        return np.log(np.abs(x * y) + 0.001) + np.cos(x + y)

class Layeb04_safe(Testfunction2D):
    """
    aka Crossfly, taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf

    Same as Layeb04, but subtracted 6 to have optimum at -6
    """
    def __init__(self):
        super().__init__(
            name='Layeb04',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=np.log(0.001)-1.-6.,
            # opt_pos can be reproduced by
            # np.unique(np.concatenate(
            #   [ [(u,v), (v,u)] for u in us for v in vs ]
            # ), axis=0)
            # where us, vs = np.zeros(4), [(2*k-1)*np.pi for k in range(-1,3)]
            opt_pos=np.array([
                ( 0.        , -9.42477796),
                ( 0.        , -3.14159265),
                ( 0.        ,  3.14159265),
                ( 0.        ,  9.42477796),
                (-9.42477796,  0.        ),
                (-3.14159265,  0.        ),
                ( 3.14159265,  0.        ),
                ( 9.42477796,  0.        ),
            ])
        )

    def objective(self, x, y):
        return np.log(np.abs(x * y) + 0.001) + np.cos(x + y) - 6.

class Layeb05(Testfunction2D):
    """
    aka Dome, taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf
    """
    def __init__(self):
        super().__init__(
            name='Layeb05',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=np.log(0.001),
            # opt_pos can be reproduced by
            # np.concatenate([
            #   [ ((2*k-1)*np.pi, 2*k*np.pi), (2*k*np.pi, (2*k-1)*np.pi) ]
            #   for k in range(-1,2)
            # ])
            opt_pos=np.array([
                (-9.42477796, -6.28318531),
                (-6.28318531, -9.42477796),
                (-3.14159265,  0.        ),
                ( 0.        , -3.14159265),
                ( 3.14159265,  6.28318531),
                ( 6.28318531,  3.14159265),
            ])
        )

    def objective(self, x, y):
        out = np.log(np.abs(np.sin(x - 0.5 * np.pi) + np.cos(y - np.pi)) + 0.001)
        out /= np.abs(np.cos(2 * x - y + 0.5 * np.pi)) + 1
        return out

class Layeb09(Testfunction2D):
    """
    taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf
    """
    def __init__(self):
        super().__init__(
            name='Layeb09',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=0.,
            # opt_pos can be reproduced by
            # np.array([
            #   (x, x) for x in [ (2*k-1) * 0.5 * np.pi for k in range(-2,4) ]
            # ])
            opt_pos=np.array([
                (-7.85398163, -7.85398163),
                (-4.71238898, -4.71238898),
                (-1.57079633, -1.57079633),
                ( 1.57079633,  1.57079633),
                ( 4.71238898,  4.71238898),
                ( 7.85398163,  7.85398163),
            ])
        )

    def objective(self, x, y):
        out = np.exp(np.abs(y * np.sin(x)) - np.abs(y)) + np.cos(x + y)
        out = np.sqrt(np.abs(out / np.exp(np.cos(x + y) - 1)))
        return out

class Layeb09_safe(Testfunction2D):
    """
    taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf

    Note: for some reason, the typed equation of the paper does not match the
          authors matlab implementation. thus the objective function refers to
          the matlab source Layeb09 from [1] > Functions > Layeb09

    [1]: https://ch.mathworks.com/matlabcentral/fileexchange/118210-new-hard-functions-for-global-optimization?s_tid=srchtitle
    """
    def __init__(self):
        super().__init__(
            name='Layeb09',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=-2.,
            # opt_pos can be reproduced by
            # np.array([
            #   (x, x) for x in [ (2*k-1) * 0.5 * np.pi for k in range(-2,4) ]
            # ])
            opt_pos=np.array([
                (-7.85398163, -7.85398163),
                (-4.71238898, -4.71238898),
                (-1.57079633, -1.57079633),
                ( 1.57079633,  1.57079633),
                ( 4.71238898,  4.71238898),
                ( 7.85398163,  7.85398163),
            ])
        )

    def objective(self, x, y):
        out = np.exp(np.abs(y * np.sin(x)) - np.abs(y)) + np.cos(x + y)
        out = np.sqrt(np.abs(out / np.exp(np.cos(x + y) - 1)))
        out -= 2.
        return out

class Layeb12(Testfunction2D):
    """
    taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf

    "This function is *very hard* to optimize.", citation from paper.
    """
    def __init__(self):
        super().__init__(
            name='Layeb12',
            area=np.array([[-5., 5.], [-5., 5.]]),
            opt_val=-(np.e+1),
            opt_pos=np.array([
                (2,  2),
            ])
        )

    def objective(self, x, y):
        out = np.cos(np.pi * (0.5 * x - 0.25 * y - 0.5))
        out = out * np.exp(np.cos(2*np.pi*x*y)) + 1
        out = -out
        return out

class Layeb12_safe(Testfunction2D):
    """
    taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf

    "This function is *very hard* to optimize.", citation from paper.

    Same as Layeb12, but subtracted 10 to have optimum at -10 - (e+1)
    """
    def __init__(self):
        super().__init__(
            name='Layeb12',
            area=np.array([[-5., 5.], [-5., 5.]]),
            opt_val=-(np.e+1)-2.,
            opt_pos=np.array([
                (2,  2),
            ])
        )

    def objective(self, x, y):
        out = np.cos(np.pi * (0.5 * x - 0.25 * y - 0.5))
        out = out * np.exp(np.cos(2*np.pi*x*y)) + 1
        out = -out - 2
        return out

class Layeb14(Testfunction2D):
    """
    taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf

    "This function is *hard* to optimize.", citation from paper.
    """
    def __init__(self):
        super().__init__(
            name='Layeb14',
            area=np.array([[-100., 100.], [-100., 100.]]),
            opt_val=0.,
            opt_pos=np.array([
                ( 0, -1),
                (-1,  0),
            ])
        )

    def objective(self, x, y):
        out = 100 * np.power(np.abs(x**2 - y - 1), 0.1)
        out += np.abs(np.log(np.power(x + y + 2, 2)))
        return out

class Layeb14_safe(Testfunction2D):
    """
    taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf

    "This function is *hard* to optimize.", citation from paper.

    same as Layeb14 but subtracted -20 to have optimum at -20
    """
    def __init__(self):
        super().__init__(
            name='Layeb14',
            area=np.array([[-100., 100.], [-100., 100.]]),
            opt_val=-20.,
            opt_pos=np.array([
                ( 0, -1),
                (-1,  0),
            ])
        )

    def objective(self, x, y):
        out = 100 * np.power(np.abs(x**2 - y - 1), 0.1)
        out += np.abs(np.log(np.power(x + y + 2, 2)))
        out -= 20.
        return out

class Layeb18(Testfunction2D):
    """
    aka Zohra, taken from "New hard benchmark functions for global optimization"
    doi: https://doi.org/10.48550/arXiv.2202.04606
    arXiv: https://arxiv.org/pdf/2202.04606.pdf
    """
    def __init__(self):
        super().__init__(
            name='Layeb18',
            area=np.array([[-10., 10.], [-10., 10.]]),
            opt_val=np.log(0.001),
            # opt_pos can be reproduced by
            # np.array([
            #   (x, x) for x in [ (2*k-1) * 0.5 * np.pi for k in range(-2,4) ]
            # ])
            opt_pos=np.array([
                (-7.85398163, -7.85398163),
                (-4.71238898, -4.71238898),
                (-1.57079633, -1.57079633),
                ( 1.57079633,  1.57079633),
                ( 4.71238898,  4.71238898),
                ( 7.85398163,  7.85398163)
            ])
        )

    def objective(self, x, y):
        out = np.log(np.abs(np.cos(2 * x * y / np.pi)) + 0.001)
        out /= np.abs(np.sin(x + y) * np.cos(x)) + 1
        return out
