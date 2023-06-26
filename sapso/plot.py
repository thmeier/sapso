import numpy as np
import matplotlib.pyplot as plt

from matplotlib import ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from . import utils
from . import sa
from . import pso

def contour_plot(history, test_func, 
                 resolution=100, ax=None, title=None, title_length='long',
                 cmap_bg='bone', cmap_fg='tab20', save=False, path=''
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

            ax.plot(*points.T, 'o', color=pick_color(particle), ms=2)

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

    if save:
        plt.savefig(path)

def comparison_plot(test_funcs, methods, kw_method, kw_plot, save=False, path=''):
    n, m = len(methods), len(test_funcs)
    fig, axes = plt.subplots(n, m, figsize=(m*2, n*2))#, layout='constrained')#, figsize=(n*2,m*2))

    for i, (method_name, kw) in enumerate(zip(methods, kw_method)):
        for j, tfunc in enumerate(test_funcs):

            if method_name == 'SA':
                method = sa.SimulatedAnnealing
            elif method_name == 'PSO':
                method = pso.ParticleSwarmOptimization
            #elif method_name == 'APSO':
            #    raise NotImplementedError('Adaptive PSO not yet implemented!')
            else:
                raise ValueError(f'Method `{method}` not recognized!')

            experiment = method(tfunc.objective, tfunc.area, **kw)
            history = experiment.optimize()
            utils.print_results(history, tfunc, end='\n\n')

            cax = contour_plot(history, tfunc, ax=axes[i, j], **kw_plot)

    cax_min, cax_max = np.min(cax.cvalues), np.max(cax.cvalues)
    cbar_ticks = [cax_min, 0.5 * (cax_min + cax_max), cax_max]

    cbar = fig.colorbar(cax, ax=axes[:,-1], ticks=cbar_ticks, pad=1.0)

    cbar.ax.set_yticklabels(
        ['min', 'median', 'max'], va='center', rotation='vertical'
    )

    xlabels = [ tfunc.name for tfunc in test_funcs ] +  [''] * (n-1) * m
    ylabels = np.array([
        [method] + [''] * (m-1) for method in methods
    ]).flatten()

    for ax, method, tfun in zip(axes.flat, xlabels, ylabels):
        # set labels only for left/bottom-most subplots
        ax.set(xlabel=method, ylabel=tfun, aspect=1.0, xticks=[], yticks=[]) 
        ax.xaxis.set_label_position('top')

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.865)

    if save:
        plt.savefig(path)
