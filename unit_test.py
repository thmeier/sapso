"""
Unit test about functionality of sapso library. 
These computations are non-sensical and only relevant for testing purposes.
"""
import sapso
import numpy as np

def main():
    #test_temperature_plot()
    #test_optimize()
    #test_testfunction_plot()
    #test_functionCallsCounterWrapper()
    test_adaptive_pso()

def test_temperature_plot():
    sapso.plot.temperature_plot()

def test_optimize():
    kwargs_sa = { 'iterations' : 100, 'seed' : 42, 'temperature' : None, 'step_size' : 0.1, 'goal' : 'min' }
    kwargs_pso = { 'iterations' : 100, 'seed' : 42, 'n_particles' : 10, 'w' : 0.75, 'a_ind' : 1., 'a_neigh' : 2., 'goal' : 'min' }

    tfunc = sapso.test.Eggholder()

    SA = sapso.sa.SimulatedAnnealing(tfunc.objective, tfunc.area, **kwargs_sa)
    PSO = sapso.pso.ParticleSwarmOptimization(tfunc.objective, tfunc.area, **kwargs_pso)

    tfunc = sapso.test.SchaffnerNo4()

    SA = sapso.sa.SimulatedAnnealing(tfunc.objective, tfunc.area, **kwargs_sa)
    PSO = sapso.pso.ParticleSwarmOptimization(tfunc.objective, tfunc.area, **kwargs_pso)

    history_sa = SA.optimize()
    history_pso = PSO.optimize()

    sapso.utils.print_results(history_sa, tfunc, end='\n\n')
    sapso.utils.print_results(history_pso, tfunc)

def test_testfunction_plot():
    tfuncs = [ 
        sapso.test.Ackley(),
        sapso.test.Beale(),
        sapso.test.CrossInTray(),
        sapso.test.Eggholder(),
        sapso.test.Himmelblau(),
        sapso.test.SchaffnerNo2() 
    ]
    # depends on test funcs
    z_offsets = [-70, -700, -70, -1100, -70, -70]
    layout = '23'

    sapso.plot.testfunction_surface_plot(
        layout, tfuncs, z_offsets,
        # edgecolor='#777', cmap_contour='bone', cmap_surface='coolwarm'
    )

    sapso.plot.testfunction_contour_plot(
        layout, tfuncs
    )

def test_functionCallsCounterWrapper():
    tfunc = sapso.test.Eggholder()

    #opt = sapso.sa.SimulatedAnnealing
    opt = sapso.pso.ParticleSwarmOptimization
    opt_args = [tfunc.objective, tfunc.area]

    fncalls_wrapper = sapso.wrappers.FunctionCallCounterWrapper
    optimizer_fncalls = fncalls_wrapper(opt, *opt_args)

    history = optimizer_fncalls.optimize()
    print('ratio optimum found / best =', history['best_val'] / tfunc.opt_val)
    print(f'with {optimizer_fncalls.total_function_calls} function calls')

def test_adaptive_pso():
    #tfunc = sapso.test.Layeb14()
    tfunc = sapso.test.Eggholder()

    apso = sapso.pso.AdaptiveParticleSwarmOptimization(
        tfunc.objective, tfunc.area,
        iterations=400,
        n_particles=200,
        components=None,        # anything else not yet supported
        interpolation='const',  # in 'cubic', 'exp', 'const'
        w=0.75,
        a_ind=1.5,
        a_neigh=2.2,
        seed=123098
      )

    history = apso.optimize()
    fitness = np.exp(tfunc.opt_val - history['best_val'])
    fitness1 = np.power(1.1, tfunc.opt_val - history['best_val'])
    print(f'best_val @ best_point = {history["best_val"]} @ {history["best_point"]}')
    print(f'opt_val @ opt_point = {tfunc.opt_val} @ {tfunc.opt_pos}')
    print(f'fitness =', fitness)
    print(f'fitness1 =', fitness1)


if __name__ == '__main__':
    h = main()
