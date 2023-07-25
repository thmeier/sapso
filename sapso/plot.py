import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from . import utils
from . import sa
from . import pso

__all__ = ['contour_plot', 'comparison_plot', 'temperature_plot']

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
    cbar = ax.contourf(X, Y, Z, cmap=cmap_bg)#, levels=resolution)

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
            history['meta']['params']['n_particles'], size=subset_size, replace=False
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

    # restrict plot limits to search space
    ax.set(xlim=area[0], ylim=area[1])

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

# TODO(low): add args and kwargs to support generic temperature functions
def temperature_plot(temperature=None, iterations=1000):
    if temperature is None:
        temperature = sa.default_temperature

    time = np.arange(1, iterations)
    temp = temperature(time, iterations)

    plt.plot(time, temp)
    plt.show()

def testfunction_surface_plot(test_funcs, z_offsets, layout, figsize,
                              resolution=100, edgecolor='#333',
                              cmap_contour='bone', cmap_surface='coolwarm',
                              show=False, save=False, path=''
                              ):

    n, m = [ int(x) for x in layout ]

    assert len(str(layout)) == 2, f'Expected 2-digit string. Got {layout}'
    assert len(test_funcs) == n * m, 'Layout and test_funcs incompatible!'

    fig =  plt.figure(figsize=figsize, dpi=200)

    for i, (test_func, z_offset) in enumerate(zip(test_funcs, z_offsets)):
        ax = fig.add_subplot(n, m, i+1, projection='3d')
        ax._axis3don = False

        objective = test_func.objective
        area = utils.validate_area(test_func.area)

        # linearly space values from lower to upper bound of corresponding dimension
        linspace_x = np.linspace(area[0, 0], area[0, 1], resolution)
        linspace_y = np.linspace(area[1, 0], area[1, 1], resolution)

        X, Y = np.meshgrid(linspace_x, linspace_y)
        Z = objective(X, Y)

        ax.plot_surface(
            X, Y, Z, edgecolor=edgecolor, cmap=cmap_surface, lw=0.5, alpha=0.75
        )

        if z_offset is None:
            z_offset = 1.25 * np.min(Z)

        ax.contourf(X, Y, Z, zdir='z', offset=z_offset, cmap=cmap_contour)

        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.gridlines.set_visible(False)
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        ax.set(
            xlim=(np.min(X), np.max(X)),
            ylim=(np.min(Y), np.max(Y)),
            zlim=(np.min(Z), np.max(Z)),
        )

        ax.set_title(test_func.name + ' Function', fontweight='bold')

    if save:
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()


def testfunction_contour_plot(test_funcs, layout, figsize,
                              resolution=100, cmap='coolwarm',
                              show=False, save=False, path=''
                              ):

    n, m = [ int(x) for x in layout ]

    assert len(str(layout)) == 2, f'Expected 2-digit string. Got {layout}'
    assert len(test_funcs) == n * m, 'Layout and test_funcs incompatible!'

    fig, axes = plt.subplots(n, m, figsize=figsize, dpi=200)

    for test_func, ax in zip(test_funcs, axes.flat):

        ax._axison = False

        objective = test_func.objective
        area = utils.validate_area(test_func.area)

        # linearly space values from lower to upper bound of corresponding dimension
        linspace_x = np.linspace(*area[0, :], resolution)
        linspace_y = np.linspace(*area[1, :], resolution)

        X, Y = np.meshgrid(linspace_x, linspace_y)
        Z = objective(X, Y)
        # contour fill
        cax = ax.contourf(X, Y, Z, cmap=cmap)
        # contour lines
        ax.contour(X, Y, Z, colors='black', alpha=0.75)
        # plot optima
        ax.plot(*test_func.opt_pos.T, '+', color='white')

        ax.set_title(test_func.name + ' Function', fontweight='bold')
        ax.set(xticks=[], yticks=[], aspect=1.0)

        # HACK FOR GENERATING REPORT
        if 'Schaffner' in test_func.name:
            ax.set(xlim=(-50,50), ylim=(-50,50))

    cax_min, cax_max = np.min(cax.cvalues), np.max(cax.cvalues)
    cbar_ticks = [cax_min, 0.5 * (cax_min + cax_max), cax_max]

    cbar = fig.colorbar(cax, ax=axes[:,-1], ticks=cbar_ticks, pad=1.0)

    cbar.ax.set_yticklabels(
        ['min', 'median', 'max'], va='center', rotation='vertical'
    )

    # make space for colorbar
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.825)

    if save:
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

def sensitivity_study_plot(test_funcs, study_config, study_results,
                           method, param, xlabel,
                           layout, figsize,
                           show=False, save=False, path=''
                           ):

    n, m = [ int(x) for x in layout ]

    assert len(str(layout)) == 2, f'Expected 2-digit string. Got {layout}'
    assert len(test_funcs) == n * m, 'Layout and test_funcs incompatible!'

    fig, axes = plt.subplots(
        n, m, figsize=figsize, dpi=200, layout='constrained'
    )

    for test_func, ax in zip(test_funcs, axes.flat):

        histories = study_results[method][param][test_func.name]['histories']

        opt_best = test_func.opt_val
        opt_found = np.array([
            h['best_val'] for h in histories.flatten()
        ]).reshape(histories.shape)

        opt_relative = opt_found / opt_best

        mean = np.mean(opt_relative, axis=-1)
        std = np.std(opt_relative, axis=-1)

        param_range = study_config[method]['params']['dynamic'][param]

        lower = np.clip(mean - std, 0.0, 1.0)
        upper = np.clip(mean + std, 0.0, 1.0)

        kw_fill = { 'alpha' : 0.25 }

        if param not in ['temperature', 'interpolation']:
            ax.plot(param_range, mean, label='mean')
            ax.fill_between(param_range, lower, upper, label='std', **kw_fill)
            ax.set_xlabel(xlabel)
        else:
            n = len(mean)
            ebar_width = 0.025
            # linearly space labels,
            # e.g.: n=2 implies [0.25, 0.75]
            #       n=3 implies [0.167, 0.5, 0.83]
            pos = np.array([ i / (2*n) for i in range(1, 2*n) ])[::2]

            errorboxes = [
                Rectangle((x - ebar_width / 2, m - s), ebar_width, 2 * s)
                for x, m, s in zip(pos, mean, std)
            ]

            # plot standard deviations
            # Create and add patch collection with specified colour/alpha
            ax.add_collection(
                PatchCollection(errorboxes, facecolor='tab:blue', alpha=0.25)
            )

            # plot means
            ax.plot(pos, mean, '_', color='tab:blue')

            ax.set(
                xlabel=xlabel[0], xlim=(0,1), xticks=pos, xticklabels=xlabel[1]
            )

        ax.set(
            ylabel='optimum : found / best', ylim=(-0.05, 1.05)
        )
        ax.set_title(test_func.name + ' Function', fontweight='bold')

    if save:
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

def benchmark_plot(bench_funcs, bench_config, bench_results, method,
                   layout, figsize, color='tab:blue',
                   show=False, save=False, path=''
                   ):

    n, m = [ int(x) for x in layout ]

    assert len(str(layout)) == 2, f'Expected 2-digit string. Got {layout}'
    assert len(bench_funcs) == n * m, 'Layout and test_funcs incompatible!'

    fig, axes = plt.subplots(
        n, m, figsize=figsize, dpi=200, layout='constrained'
    )

    for bench_func, ax in zip(bench_funcs, axes.flat):

        # fitness w.r.t. bench_func
        fitness = lambda xs: np.exp(bench_func.opt_val - xs)
        # all histories, i.e. one per run
        histories = bench_results[method][bench_func.name]['histories']
        # all encountered values per run
        values = np.array([ h['values'] for h in histories ])
        # best encountered values per run
        optima = np.array([ h['best_val'] for h in histories ])
        # fitness of the encountered values
        fitness_values = fitness(values)
        fitness_optima = fitness(optima)
        # corresponding number of fitness calls
        fncalls_values = np.arange(values.shape[1])
        fncalls_optima = np.array([
            np.argmax(vs == os) for vs, os in zip(values, optima)
        ])

        # plot best attainable fitness
        ax.axhline(
            1.0, xmin=0, xmax=1, ls=':', color='gray', ms=1, alpha=0.75,
        )

        # for all n_retries, plot the fitness of the found values
        for _fitness in fitness_values:
            ax.plot(
                fncalls_values, _fitness, '+', color=color, ms=1, alpha=0.5,
            )

        # for all n_retries, plot the fitness of the best encountered values
        ax.plot(
            fncalls_optima, fitness_optima, 'x', color=color, ms=2, alpha=1.0,
        )

        # mean and std of fitness of all best encountered values
        mean_optima, std_optima = np.mean(fitness_optima), np.std(fitness_optima)

        # plot std of fitness of all best encountered values as fat hline
        ax.add_collection(PatchCollection([Rectangle(
            (fncalls_values[0], mean_optima - std_optima), # lower left
            fncalls_values[-1], 2 * std_optima             # width, height
        )], facecolor=color, alpha=0.25))

        # plot mean of fitness all best encountered values as hline
        ax.axhline(
            mean_optima, color=color, ls='--', alpha=1.0
        )

        # adjust plot
        ax.set_title(bench_func.name + ' Function', fontweight='bold')
        ax.set(
            xlim=(fncalls_values[0], fncalls_values[-1]), ylim=(0.0, 1.05),
            xlabel='# fitness calls', ylabel='bad       fitness       good'
        )
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f'{int(x // 10000)}E4')
        )


    if save:
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()
