import numpy as np

"""
Perform simulated annealing optimization on objective functi
"""
def simulated_annealing(
        objective, area,
        temperature=None,
        iterations=5000,
        step_size=0.1,
        seed=42,
        goal='min'
        ):

    # SETUP #------------------------------------------------------------------#

    # ensure working with np.ndarray
    if not isinstance(area, np.ndarray):
        area = np.array(area)

    # extract dimensions
    n, m = area.shape

    # assert correct number of constraints
    assert m == 2, (f'The bounding hyper cube `area` of the search space '
                    f'has inavlid dimensions. Expected (n, 2) got {area.shape}')

    del m # not needed, delete to reduce name clashes

    # use default temperature cooling scheme if not specified
    if temperature is None:
        def temperature(it_curr, it_max=iterations, temp_max=0.1):
            return temp_max * (1 - it_curr / it_max)

    # assert correct optimization goal
    assert goal.lower() in ['min', 'max'], (f'Invalid optimization goal `goal`. '
                                            f'Expected "min" or "max" got {goal}')

    # define better in terms of optimization goal
    if goal.lower() == 'min':
        better = lambda x, y: x < y
    else:
        better = lambda x, y: x > y

    # INIT #-------------------------------------------------------------------#

    # set seed for reproducibility
    np.random.seed(seed)

    # create and evaluate initial point inside bounding hyper cube `area`
    p_curr = area[:, 0] + np.random.rand(n) * (area[:, 1] - area[:, 0])
    f_curr = objective(*p_curr)

    # track best candidate and its value
    p_best, f_best = p_curr, f_curr

    # track history of encountered points
    history = {'points': [], 'values': []}

    # SA #---------------------------------------------------------------------#

    for i in range(1, iterations):

        # UPDATE #-------------------------------------------------------------#

        # for each dimension,
        # choose a direction between -1 and 1 and scale by step_size
        p_delta = (np.random.rand(n) * 2 - np.ones(n)) * step_size

        # TODO: check if point update rule makes sense
        #       i.e. can at most make steps of size `step_size`
        # TODO: make sure new point is still inside `area`
        #       of course for our goals this isn't an issue
        #       BUT the user might expect points to stay inside the specified region
        #       because for example due to external constraints

        # calculate and evaluate new candidate point
        p_new = p_curr + p_delta
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
        'points': np.array(history['points']),
        'values': np.array(history['values']),
        'p_best': p_best,
        'f_best': f_best,
    }

    # return best encountered point, value and history of points
    return history
