import numpy as np

from functools import partial
from . import utils
from . import base


def default_temperature(it_curr, it_max, temp_max=0.1):
    return max(temp_max * (1 - it_curr / it_max), 1e-7)

class SimulatedAnnealing(base.OptimizationMethod):
    def __init__(self, 
                 objective, area,
                 iterations=100,
                 seed=42,
                 temperature=None,
                 step_size=0.1,
                 goal='min'
                 ):
        # f
        super().__init__(objective, area, iterations, seed, goal)

        # use default temperature cooling scheme if not specified
        if temperature is None:
            temperature = partial(default_temperature, it_max=iterations)

        self.params = base.Params()
        self.params.temperature = temperature
        self.params.step_size = step_size # TODO: validate step_size in [0,1]

        # make step area-independent
        self.params.step = self.params.step_size * np.min(np.diff(self.area))

    def _reset_history(self):
        # track history of encountered points
        self.history = {'points': [], 'values': []}

    #TODO: rename input variables
    def _update_history(self, p_new, f_new):
        # keep track of point and value
        self.history['points'].append(p_new)
        self.history['values'].append(f_new)

    def _finalize_history(self):
        # ensure the history uses numpy arrays, facilitates plotting with matplotlib
        self.history = {
            # results
            'points'     : np.array(self.history['points']),
            'values'     : np.array(self.history['values']),
            'best_val'   : self.f_best, # TODO: rename to best_val
            'best_point' : self.p_best, # TODO: rename to best_pos
            # meta information
            'meta'       : {
                # TODO: remove method, rename method_short to method
                'method'       : 'simulated_annealing',
                'method_short' : 'SA',
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
        p_curr = utils.uniform(self.area)
        f_curr = self.objective(*p_curr)

        # TODO RENAME
        # track best candidate and its value
        self.p_best, self.f_best = p_curr, f_curr

        self._reset_history()

        for i in range(1, self.iterations+1):

            # TODO: use self.rng.uniform?
            # for each dimension,
            # choose a direction between -1 and 1 and scale by step
            p_delta = (np.random.rand(self.n) * 2 - np.ones(self.n)) * self.params.step

            # clips i-th value into interval area[i, :]
            p_new = np.clip(p_curr + p_delta, *self.area.T)

            f_new = self.objective(*p_new)

            self._update_history(p_new, f_new)

            # local improvement?
            if np.less(f_new, f_curr):

                # remember locally best point and value
                p_curr, f_curr = p_new, f_new

                # global improvement?
                if np.less(f_new, self.f_best):

                    # remember globally best point and value
                    self.p_best, self.f_best = p_new, f_new

            # TODO: make use of rng
            # accept worse point with temperatue-dependent Boltzmann-like probability
            elif np.random.rand() < np.exp(-np.abs(f_new - self.f_best) / self.params.temperature(i)):

                # remember worse point and value
                p_curr, f_curr = p_new, f_new

        self._finalize_history()

        return self.history

# NOTE: LEGACY CODE - WILL BE REMOVED
#def simulated_annealing(
#        objective, area,
#        iterations=5000,
#        seed=42,
#        temperature=None,
#        step_size=0.1,
#        goal='min'
#        ):
#    """
#    Perform simulated annealing optimization on objective function
#
#    input
#    -----
#    objective : function
#        Objective/fitness/cost function, i.e. $\R^n \mapsto \R$,
#        where $n$ is the dimensionality of the optimization problem.
#
#    area : numpy.ndarray
#        Bounding hypercube of search space, i.e. has shape $(n, 2)$,
#        where $n$ is the dimensionality of the optimization problem.
#        Thus `area` is an array of lower and upper bounds, which equivalently
#        means, `area[i] == [i_min, i_max]`, where `i_min` and `i_max` 
#        denote the lower and upper bound of the i-th component.
#
#    temperature : function | None
#
#    # TODO: finish documentation
#    """
#
#    # SETUP #------------------------------------------------------------------#
#
#    # validates if area is in correct format
#    area = utils.validate_area(area)
#
#    # extract dimensions
#    n, _ = area.shape
#
#    # use default temperature cooling scheme if not specified
#    if temperature is None:
#        temperature = partial(default_temperature, it_max=iterations)
#
#    goal = utils.validate_goal(goal)
#    better, _, _ = utils.comparison_funcs_from_goal(goal)
#
#    # INIT #-------------------------------------------------------------------#
#
#    # set seed for reproducibility
#    np.random.seed(seed)
#
#    # make step area-independent
#    step = step_size * np.min(np.diff(area))
#
#    # create and evaluate initial point inside bounding hyper cube `area`
#    p_curr = utils.uniform(area)
#    f_curr = objective(*p_curr)
#
#    # track best candidate and its value
#    p_best, f_best = p_curr, f_curr
#
#    # track history of encountered points
#    history = {'points': [], 'values': []}
#
#    # SA #---------------------------------------------------------------------#
#
#    for i in range(1, iterations+1):
#
#        # UPDATE #-------------------------------------------------------------#
#
#        # for each dimension,
#        # choose a direction between -1 and 1 and scale by step
#        p_delta = (np.random.rand(n) * 2 - np.ones(n)) * step
#
#        # calculate and evaluate new candidate point
#        p_new = p_curr + p_delta
#        # clips i-th value of p_new into interval area[i, :]
#        p_new = np.clip(p_new, *area.T)
#
#        f_new = objective(*p_new)
#
#        # keep track of point and value
#        history['points'].append(p_new)
#        history['values'].append(f_new)
#
#        # local improvement?
#        if better(f_new, f_curr):
#
#            # remember locally best point and value
#            p_curr, f_curr = p_new, f_new
#
#            # global improvement?
#            if better(f_new, f_best):
#
#                # remember globally best point and value
#                p_best, f_best = p_new, f_new
#
#        # accept worse point with temperatue-dependent Boltzmann-like probability
#        elif np.random.rand() < np.exp(-np.abs(f_new - f_best) / temperature(i)):
#
#            # remember worse point and value
#            p_curr, f_curr = p_new, f_new
#
#    # ensure the history uses numpy arrays, facilitates plotting with matplotlib
#    history = {
#        # results
#        'points'     : np.array(history['points']),
#        'values'     : np.array(history['values']),
#        'best_point' : p_best,
#        'best_val'   : f_best,
#        # meta information
#        'meta'       : {
#            'method'       : 'simulated_annealing',
#            'method_short' : 'SA',
#            'params'       : {
#                'iterations' : iterations,
#                'seed'       : seed,
#                'goal'       : goal,
#                'step_size'  : step_size,
#                'step'       : step
#            },
#        },
#    }
#
#    # return best encountered point, value and history of points
#    return history
