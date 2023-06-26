import numpy as np

from functools import partial

from . import utils
from . import base


def default_temperature(it_curr, it_max, temp_max=0.1):
    return max(temp_max * (1 - it_curr / it_max), 1e-7)

class SimulatedAnnealing(base.OptimizationMethod):
    """
    Perform simulated annealing optimization on objective function

    input
    -----
    objective : function
        Objective/fitness/cost function, i.e. $\R^n \mapsto \R$,
        where $n$ is the dimensionality of the optimization problem.

    area : numpy.ndarray
        Bounding hypercube of search space, i.e. has shape $(n, 2)$,
        where $n$ is the dimensionality of the optimization problem.
        Thus `area` is an array of lower and upper bounds, which equivalently
        means, `area[i] == [i_min, i_max]`, where `i_min` and `i_max` 
        denote the lower and upper bound of the i-th component.

    temperature : function | None

    # TODO(low): finish SA documentation
    """
    def __init__(self, 
                 objective, area,
                 iterations=1000,
                 seed=42,
                 temperature=None,
                 step_size=0.1,
                 goal='min'
                 ):
        super().__init__(objective, area, iterations, seed, goal)

        # use default temperature cooling scheme if not specified
        if temperature is None:
            temperature = partial(default_temperature, it_max=iterations)

        self.params = base.Params()
        self.params.temperature = temperature
        self.params.step_size = step_size

        # make step area-independent
        self.params.step = self.params.step_size * np.min(np.diff(self.area))

    def _reset_history(self):
        # track history of encountered points
        self.history = {'points': [], 'values': []}

    def _update_history(self, pos, val):
        # keep track of point and value
        self.history['points'].append(pos)
        self.history['values'].append(val)

    def _finalize_history(self):
        # ensure the history uses numpy arrays, facilitates plotting with matplotlib
        self.history = {
            # results
            'points'     : np.array(self.history['points']),
            'values'     : np.array(self.history['values']),
            'best_val'   : self.best_val,
            'best_point' : self.best_pos,
            # meta information
            'meta'       : {
                'method'       : 'SA',
                'params'       : {
                    'iterations' : self.iterations,
                    'seed'       : self.seed,
                    'goal'       : self.goal,
                    'step_size'  : self.params.step_size,
                    'step'       : self.params.step
                },
            },
        }

    def optimize(self):

        # create and evaluate initial point inside bounding hyper cube `area`
        curr_pos = utils.uniform(self.rng, self.area)
        curr_val = self.objective(*curr_pos)

        self.best_pos, self.best_val = curr_pos, curr_val

        self._reset_history()

        for i in range(self.iterations):

            # choose a random direction between -1 and 1 and scale by step
            delta = (self.rng.random(self.n) * 2 - np.ones(self.n)) * self.params.step

            # clips k-th value into interval area[k, :]
            new_pos = np.clip(curr_pos + delta, *self.area.T)

            new_val = self.objective(*new_pos)

            self._update_history(new_pos, new_val)

            # local improvement?
            if np.less(new_val, curr_val):

                # remember locally best point and value
                curr_pos, curr_val = new_pos, new_val

                # global improvement?
                if np.less(new_val, self.best_val):

                    # remember globally best point and value
                    self.best_pos, self.best_val = new_pos, new_val

            elif self.rng.random() < np.exp(-np.abs(new_val - self.best_val) / self.params.temperature(i+1)):

                # remember worse point and value
                curr_pos, curr_val = new_pos, new_val

        self._finalize_history()

        return self.history
