"""
# abstract SA and PSO algorithm into a single super class 
# this allows to implement SAPSO the combination using the same super class
# and simply the plotting and keeping track of history points in a uniform way
"""

# abstract base class for implementing interfaces
from abc import ABC, abstractmethod 

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

    plot : function
        polymorphically plot the results given the history returned by optimize
    """

    def __init__(objective, area, iterations, seed):
        self.objective = objective
        self.area = area
        self.iterations = iterations
        self.seed = seed

    @abstractmethod
    def setup():
        pass

    @abstractmethod
    def optimize():
        pass

    def plot():
        history = self.optimize()
        # do the plotting depending on history algo type etc
