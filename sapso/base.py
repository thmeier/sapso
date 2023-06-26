"""
# abstract SA and PSO algorithm into a single super class 
# this allows to implement SAPSO the combination using the same super class
# and simply the plotting and keeping track of history points in a uniform way
"""

import numpy as np

from abc import ABC, abstractmethod 

from . import utils

class OptimizationMethod(ABC):
    """
    Abstract base class (ABC) that specifies the interface for an optimization
    method. This interface allows to nicely compare different methods such as
    `simulated_annealing` and `particle_swarm_optimization`.

    methods
    -------
    setup : abstractmethod
        prepare all variables and constants before actually optimizing

    optimize : abstractmethod
        perform actual optimization
    """

    def __init__(self, objective, area, iterations, seed, goal='min'):
        self.goal = utils.validate_goal(goal)
        # NOTE: maximization is equivalent to minimization of the negative cost
        if self.goal == 'max':
            objective = lambda *args, **kwargs: -objective(*args, **kwargs)

        self.objective = objective
        self.area = utils.validate_area(area)
        # dimensionality of optimization problem
        self.n = self.area.shape[0]
        self.iterations = iterations
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.history = dict()

    @abstractmethod
    def _reset_history():
        pass

    @abstractmethod
    def _update_history():
        pass

    @abstractmethod
    def _finalize_history():
        pass

    @abstractmethod
    def optimize():
        pass


class Params(object):
    """
    Object to hold arbitrary attributes with description.

    usage
    -----

    ```
    params = Params('string to be used as title')
    params
    > 'string to be used as title'
    params.any_atttribute = 42
    params.another_config_attr = [777, 'nice']
    ```
    """

    def __init__(self, *args):
        self.__header__ = str(args[0]) if args else None

    def __repr__(self):
        if self.__header__ is None:
             return super(Params, self).__repr__()
        return self.__header__

    def __next__(self):
        """ Fake iteration functionality.
        """
        raise StopIteration

    def __iter__(self):
        """ Fake iteration functionality.
        We skip magic attribues and Structs, and return the rest.
        """
        ks = self.__dict__.keys()
        for k in ks:
            if not k.startswith('__') and not isinstance(k, Params):
                yield getattr(self, k)

    def __len__(self):
        """ Don't count magic attributes or Structs.
        """
        ks = self.__dict__.keys()
        return len([k for k in ks if not k.startswith('__')\
                    and not isinstance(k, Params)])

