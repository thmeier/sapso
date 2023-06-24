import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker

from . import utils

def contour_plot(history, test_func, 
                 resolution=100, ax=None, title=None, title_length='long',
                 cmap_bg='bone', cmap_fg='tab20'
                 ):
    """
    Plots a 2D contour plot of the objective function `objective` over the region `area`.
    """

    # SETUP #------------------------------------------------------------------#

    objective = test_func.objective
    area = utils.validate_area(test_func.area)

    # do we have user-specified title?
    got_title = title is not None

    # set title automatically if empty
    if title == '':
        title = utils.title_from_history(history, test_func, title_length)

    # INIT #-------------------------------------------------------------------#

    # linearly space values from lower to upper bound of corresponding dimension
    linspace_x = np.linspace(area[0, 0], area[0, 1], resolution)
    linspace_y = np.linspace(area[1, 0], area[1, 1], resolution)

    X, Y = np.meshgrid(linspace_x, linspace_y)
    Z = objective(X, Y)

    # PLOT #-------------------------------------------------------------------#

    # do we have user-specified axis?
    got_axis = ax is not None

    # create fig and axis if not specified by user
    if not got_axis:
        fig, ax = plt.subplots(1, 1)

    # add 2D contour plot
    cbar = ax.contourf(X, Y, Z, cmap=cmap_bg, levels=resolution)

    # add colorbar if didn't get axis
    if not got_axis:
        ax.figure.colorbar(cbar)

    if history['meta']['method'] == 'SA':
        # add all encountered points
        ax.plot(*history['points'].T, '-', color='tab:orange')

    elif history['meta']['method'] == 'PSO':
        # randomly select a subset of particles to plot
        subset_size = 3
        selected = np.random.choice(
            history['meta']['params']['n_particles'], size=subset_size, replace=True
        )

        # get colors from nice qualitative cmap
        colors = utils.colors_from_cmap(cmap_fg)
        pick_color = lambda i: colors[i % len(colors)]

        # for each tracked particle, plot its positions
        for particle in selected:
            mask = history['particle_id'] == particle
            points = history['points'][mask]

            ax.plot(*points.T, 'o', color=pick_color(particle))

            #print(f'VERBOSE: history["particle_id"].shape = {history["particle_id"].shape}')
            #print(f'VERBOSE: history["points"].shape = {history["points"].shape}')
            #print(f'VERBOSE: mask.shape = {mask.shape}, any = {np.any(mask)}')
            #print(f'VERBOSE: points.shape = {points.shape}')

            #print(f'VERBOSE: history["points"] = \n{history["points"]}')
            #print(f'VERBOSE: particle id = {particle}')
            #print(f'VERBOSE: points = \n{points[:100,:]}')

    else:
        raise RuntimeError(
            f"Plotting for method `{history['meta']['method']}` not supported!"
        )

    # add best encountered point
    ax.plot(*history['best_point'], 'o', color='white')

    # add optimal points
    ax.plot(*test_func.opt_pos.T, '+', color='red')

    # set title if specified
    if got_title:
        ax.set_title(title) #, pad=9.0)

    # set labels if didn't get axis
    if not got_axis:
        ax.set(xlabel='x', ylabel='y')

    # let caller handle plotting of colorbar if the axis was specified
    if got_axis:
        return cbar
