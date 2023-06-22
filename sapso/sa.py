import utils
import numpy as np

# TODO: scale step_size by area lengths (c.f. PSO)
def simulated_annealing(
        objective, area,
        temperature=None,
        iterations=5000,
        step_size=0.1,
        seed=42,
        goal='min'
        ):
    """
    Perform simulated annealing optimization on objective function

    input
    -----
    objective : function
        objective/fitness/cost function, i.e. $\R^n \mapsto \R$,
        where $n$ is the dimensionality of the optimization problem.

    area : numpy.ndarray
        bounding hypercube of search space, i.e. has shape $(n, 2)$,
        where $n$ is the dimensionality of the optimization problem.
        Thus `area` is an array of lower and upper bounds, which equivalently
        means, `area[i] == [i_min, i_max]`, where `i_min` and `i_max` 
        denote the lower and upper bound of the i-th component.

    temperature : function | None

    # TODO: finish documentation
    """

    # SETUP #------------------------------------------------------------------#

    # validates if area is in correct format
    area = utils.validate_area(area)

    # extract dimensions
    n, _ = area.shape

    # use default temperature cooling scheme if not specified
    if temperature is None:
        def temperature(it_curr, it_max=iterations, temp_max=0.1):
            return temp_max * (1 - it_curr / it_max)

    goal = validate_goal(goal)
    better, _, _ = utils.comparison_funcs_from_goal(goal)

    # INIT #-------------------------------------------------------------------#

    # set seed for reproducibility
    np.random.seed(seed)

    # create and evaluate initial point inside bounding hyper cube `area`
    p_curr = utils.uniform(area)
    f_curr = objective(*p_curr)

    # track best candidate and its value
    p_best, f_best = p_curr, f_curr

    # track history of encountered points
    history = {'points': [], 'values': []}

    # SA #---------------------------------------------------------------------#

    for i in range(1, iterations+1):

        # UPDATE #-------------------------------------------------------------#

        # for each dimension,
        # choose a direction between -1 and 1 and scale by step_size
        p_delta = (np.random.rand(n) * 2 - np.ones(n)) * step_size

        # TODO: check if point update rule makes sense
        #       i.e. can at most make steps of size `step_size`
        #       i.e. make step-size dependent on area

        # calculate and evaluate new candidate point
        p_new = p_curr + p_delta
        # clips i-th value of p_new into interval area[i, :]
        p_new = np.clip(p_new, *area.T)

        f_new = objective(*p_new)

        # keep track of point and value
        history['points'].append(p_new)
        history['values'].append(f_new)

        # local improvement?
        if better(f_new, f_curr):

            # remember locally best point and value
            p_curr, f_curr = p_new, f_new

            # global improvement?
            if better(f_new, f_best):

                # remember globally best point and value
                p_best, f_best = p_new, f_new

        # accept worse point with temperatue-dependent Boltzmann-like probability
        elif np.random.rand() < np.exp(-np.abs(f_new - f_best) / temperature(i)):

            # remember worse point and value
            p_curr, f_curr = p_new, f_new

    # ensure the history uses numpy arrays, facilitates plotting with matplotlib
    history = {
        # results
        'points'     : np.array(history['points']),
        'values'     : np.array(history['values']),
        'best_point' : p_best,
        'best_val'   : f_best,
        # meta information
        'algorithm'  : 'simulated_annealing',
        'params'     : {
            'goal'       : goal,
            'seed'       : seed,
            'iterations' : iterations,
            'step_size'  : step_size
        },
    }

    # return best encountered point, value and history of points
    return history
