"""
Utility functions that abstract away unnecessary details.
"""

import numpy as np
import matplotlib as mpl


def normalize(v):
    """
    Numerically stable vector normalization.

    input
    -----
    v : numpy.ndarray
        A single vector of shape $(n,)$

    output
    ------
    n : numpy.ndarray
        The (approximately) normalized vector
    """
    n = v / (np.linalg.norm(v) + 1e-7)
    return n

# TODO: check with rng.uniform(low=,high=,size=)
def uniform(area):
    """
    Compute a n-dimensional random vector,
    where each component is uniformly distributed
    between the bounding interval of the hypercube.

    input
    -----
    area : numpy.ndarray
        n-dimensional bounding hypercube 
        has shape $(n, 2)$

    output
    ------
    u : numpy.ndarray
        has shape $(n,)$
        where the i-th component is uniformly distributed between area[i, :]
    """
    u = area[:, 0] + np.random.rand(area.shape[0]) * (area[:, 1] - area[:, 0])
    return u

def validate_area(area):
    """
    Validates that input really is a n-dimensional bounding hypercube.

    input
    -----
    area : numpy.ndarray | list of lists
        n-dimensional bounding hypercube 
        has shape $(n, 2)$

    output
    ------
    area : numpy.ndarray
        n-dimensional bounding hypercube 
        has shape $(n, 2)$
    """
    # ensure working with np.ndarray
    if not isinstance(area, np.ndarray):
        area = np.array(area)

    # extract dimensions
    n, m = area.shape

    # assert correct number of constraints
    assert m == 2, (
        f'The bounding hyper cube `area` of the search space '
        f'has inavlid dimensions. Expected (n, 2) got {area.shape}'
    )

    # assert lower is smaller equal to upper bound
    assert np.all(np.diff(area) > 0) , (
        f'The bounding hyper cuber `area` of the search space '
        f'has invalid bounds. Expected `area[:, 0] <= area [:, 1]`'
    )

    return area

def validate_goal(goal):
    """
    Validates that input really is a valid optimization goal.

    input
    -----
    goal : string
        One of `min` or `max`, denotes wheter minimization or maximization is sought

    output
    ------
    goal : string
        One of `min` or `max`, denotes wheter minimization or maximization is sought
    """
    goal = goal.lower()

    # assert valid optimization goal
    assert goal in ['min', 'max'], (
        f'Invalid optimization goal `goal`. Expected "min" or "max" got {goal}'
    )

    return goal

def comparison_funcs_from_goal(goal):
    """
    Return appropriate comparison functions depending on a valid optimization goal

    input
    -----
    goal : string
        One of `min` or `max`, denotes wheter minimization or maximization is sought

    output
    ------
    better : function
        Decides whether input `x` is better than input `y`
        i.e. one of `<` and `>` depending on goal.

    extremum : function
        numpy function. One of `np.max` and `np.min`, depending on `goal`

    argextremum : function
        numpy functin. One of `np.argmax` and `np.argmin` depending on `goal`
    """

    # define better in terms of optimization goal
    if goal.lower() == 'min':
        better, extremum, argextremum = lambda x, y: x < y, np.min, np.argmin
    else:
        better, extremum, argextremum = lambda x, y: x > y, np.max, np.argmax

    return better, extremum, argextremum

def title_from_history(history, test_func, title_length='long'):
    testf      = test_func.name
    method     = history['meta']['method']
    params     = history['meta']['params']
    iterations = params['iterations']
    seed       = params['seed']

    info_method = f'{method} @ {testf}\n'
    info_general = f'it:{iterations}, seed:{seed}'

    if method == 'PSO':
        con = '\n'
        info_specific =  f"w: {params['w']}, "
        info_specific += f"a_ind: {params['a_ind']}, "
        info_specific += f"a_neigh: {params['a_neigh']}"

    elif method == 'SA':
        con = ', '
        info_specific = f"step_size: {params['step_size']}"

    else:
        raise RuntimeError(f'Method `{method}` not supported!')

    if title_length == 'long':
        title = info_method + info_general + con + info_specific

    elif title_length == 'short':
        title = info_method

    else:
        raise RuntimeError(f'Title length `{title_length}` not supported!')

    return title

def print_results(history, test_func, end='\n'):
    pos_str = ''
    for pos in test_func.opt_pos:
        pos_str += f'{pos} | '
    pos_str = pos_str.rstrip(' | ')

    print(f"optimum - {history['meta']['method']} @ {test_func.name}:", end='\n\n')
    print(f"* found    : pos: {history['best_point']}")
    print(f"           : val: {history['best_val']}", end='\n\n')
    print(f"* expected : pos: {pos_str}")
    print(f"           : val: {test_func.opt_val}", end=end)

def colors_from_cmap(cmap_name):
    """
    Get the individual colors of qualitative matplotlib colormaps, c.f. [1].
    This facilitates picking a large number of colors that are still pleasant.

    [1]: https://matplotlib.org/stable/tutorials/colors/colormaps.html#qualitative
    """
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors = np.unique(np.array([
        cmap(i) for i in np.linspace(0,1, 100)
    ]), axis=0)
    return colors
