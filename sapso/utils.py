"""
Utility functions that abstract away unnecessary details.
"""

import numpy as np


def normalize(v):
    """
    Numerically stable vector normalization.

    input
    -----
    v : numpy.ndarray
        a single vector of shape $(n,)$

    output
    ------
    n : numpy.ndarray
        the (approximately) normalized vector
    """
    n = v / (np.linalg.norm(v) + 1e-7)
    return n

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
        from $R^2$ to ${true, false}$
        decides whether input `x` is better than input `y`
        i.e. one of `<` and `>`

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
