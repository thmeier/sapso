import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from . import utils

def contour_plot(objective, area, history, resolution=100, ax=None, title=''):
    """
    Plots a 2D contour plot of the objective function `objective` over the region `area`.
    """

    # SETUP #------------------------------------------------------------------#

    area = utils.validate_area(area)

    # INIT #-------------------------------------------------------------------#

    # linearly space values from lower to upper bound of corresponding dimension
    linspace_x = np.linspace(area[0, 0], area[0, 1], resolution)
    linspace_y = np.linspace(area[1, 0], area[1, 1], resolution)

    X, Y = np.meshgrid(linspace_x, linspace_y)
    Z = objective(X, Y)

    # PLOT #-------------------------------------------------------------------#

    # do we have user-specified axis?
    got_axis = ax is not None

    # create fig and axis if not specified
    if not got_axis:
        fig , ax = plt.subplots(1,1)

    # add 2D contour plot
    cbar = ax.contourf(X, Y, Z, locator=ticker.LogLocator(), levels = resolution)

    # add colorbar if didn't get axis
    if not got_axis:
        ax.figure.colorbar(cbar)

    # add all encountered points
    ax.plot(*history['points'].T, '-', color='tab:orange')

    # add best encountered point
    ax.plot(*history['best_point'], 'o', color='white')

    # set title if specified
    if title != '':
        ax.set_title(title)

    # set labels if didn't get axis
    if not got_axis:
        ax.set(xlabel='x', ylabel='y')

    # let caller handle plotting of colorbar if the axis was specified
    if got_axis:
        return cbar
