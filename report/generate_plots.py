import os
import sys
sys.path.append('/'.join(__file__.split('/')[:-2]))  # extend PATH for sapso

import sapso
import numpy as np

def main():

    #
    # SETTINGS
    #

    # matplotlib
    figsize = (13, 6.5)  # power point has slide dimensions (13.33, 7.5)
    layout = '23'        # plots in 2 x 3 grid

    # sensitivity study & benchmark
    master_seed = 43            # seed for seed generation from seed pool
    n_retries_sensitivity = 10  # number of retires for examining the sensitivity of a single parameter
    n_retries_benchmark = 20    # number of retries for benchmark experiments
    max_seed = 75385132812489   # DO NOT CHANGE - max seed of seed pool

    # experiments
    TEST_FUNCTION_CONTOUR_PLOT = False
    TEST_FUNCTION_SURFACE_PLOT = False
    SENSITIVITY_STUDY = False
    SENSITIVITY_STUDY_PLOT = False

    BENCHMARK_FUNCTION_CONTOUR_PLOT = False
    BENCHMARK_FUNCTION_SURFACE_PLOT = False
    BENCHMARK = True
    BENCHMARK_PLOT = True

    test_functions = [
        sapso.test.Ackley_safe(),        # easy
        sapso.test.Eggholder(),          # easy
        sapso.test.SchaffnerNo2_safe(),  # easy
        sapso.test.BukinNo6_safe(),      # easy
        sapso.test.HoelderTable(),       # easy
        sapso.test.Michalewicz(),        # easy
    ]

    # test function surface plot z offsets
    z_offsets_test = [
        -70, -700, -70, -1100, -70, -70
    ]

    benchmark_functions = [
        sapso.test.Layeb04(),
        sapso.test.Layeb05(),
        sapso.test.Layeb09(),
        sapso.test.Layeb12(),  # very hard
        sapso.test.Layeb14(),  # hard
        sapso.test.Layeb18(),
    ]

    # test function surface plot z offsets
    z_offsets_benchmark = [
        None, None, -10.5, None, 50, -6
    ]

    #
    # TEST FUNCTION VISUALIZATION
    #

    if TEST_FUNCTION_CONTOUR_PLOT:
        sapso.plot.testfunction_contour_plot(
            test_functions, layout, figsize, show=False, save=True,
            path='overview_testfunction_contour.pdf'
        )

    if TEST_FUNCTION_SURFACE_PLOT:

        sapso.plot.testfunction_surface_plot(
            test_functions, z_offsets_test, layout, figsize,
            show=False, save=True,
            path='overview_testfunction_surface.pdf'
        )

    #
    # SENSITIVITY STUDY
    #

    if SENSITIVITY_STUDY:

        study_config = {
            'SA' : {
                'implementation' : sapso.sa.SimulatedAnnealing,
                'params' : {
                    'fix' : {
                        'iterations'  : 2000,
                        'step_size'   : 0.2,
                        'temperature' : sapso.sa.temperature_exp,
                    },
                    'dynamic' : {
                        'iterations'  : np.array([1000, 1500, 2000, 3000, 5000]),
                        'step_size'   : np.array([0.1, 0.2, 0.3, 0.4]),
                        # negligible effect
                        # 'temperature' : [sapso.sa.temperature_lin, sapso.sa.temperature_exp],
                    },
                    'labels' : [
                        '# iterations',
                        'step size (in % rel. search space)',
                        # negligible effect
                        # ('temperature', ['lin. decay', 'exp. decay'])
                    ]
                },
            },
            'PSO' : {
                'implementation' : sapso.pso.ParticleSwarmOptimization,
                'params' : {
                    'fix' : {
                        'iterations'  : 500,
                        'n_particles' : 200,
                        'w'           : 0.75,
                        'a_ind'       : 1.0,
                        'a_neigh'     : 1.6,
                    },
                    'dynamic' : {
                        'iterations'  : np.array([200, 300, 400, 500, 750]),
                        'n_particles' : np.array([100, 200, 300]),
                        'w'           : np.array([0.6, 0.7, 0.8]),
                        # negligible effect
                        # 'a_ind'       : np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                        # negligible effect
                        # 'a_neigh'     : np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                    },
                    'labels' : [
                        '# iterations',
                        'number of particles',
                        'inertia weight w',
                        # negligible effect
                        # 'acceleration (individual optimum)',
                        # negligible effect
                        # 'acceleration (swarm optimum)',
                    ]
                },
            },
            'APSO' : {
                'implementation' : sapso.pso.AdaptiveParticleSwarmOptimization,
                'params' : {
                    'fix' : {
                        'iterations'    : 500,
                        'n_particles'   : 200,
                        'w'             : 0.75,
                        'a_ind'         : 1.0,
                        'a_neigh'       : 2.0,
                        'interpolation' : 'const',
                    },
                    'dynamic' : {
                        'iterations'    : np.array([200, 500, 1000, 2500, 3500, 5000]),
                        'n_particles'   : np.array([100, 200, 300, 400]),
                        'w'             : np.array([0.5, 0.6, 0.8, 0.9]),
                        'a_ind'         : np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                        'a_neigh'       : np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
                        'interpolation' : ['const', 'cubic', 'exp'],
                    },
                    'labels' : [
                        '# iterations',
                        'number of particles',
                        'inertia weight w',
                        'acceleration (individual optimum)',
                        'acceleration (swarm optimum)',
                        ('adaptiveness', ['const', 'cubic', 'exp'])
                    ]
                },
            },
        }

        n_methods = len(study_config.keys())
        n_seeds = n_methods * n_retries_sensitivity

        assert max_seed >= n_seeds, f'Seed pool too small. {max_seed} < {n_seeds}'

        master_rng = np.random.default_rng(seed=master_seed)
        seed_pool = master_rng.choice(
            max_seed, size=n_seeds, replace=False
        ).reshape(n_methods, n_retries_sensitivity)

        # holding all the results
        study_results = dict()

        print('* begin sensitivity study')

        for method_name, seeds in zip(study_config.keys(), seed_pool):
            print(f'  * {method_name}')

            # optimization method, i.e. SA, PSO, APSO
            opt = study_config[method_name]['implementation']
            # fixed params
            params_fix = study_config[method_name]['params']['fix']
            # params under sensitivity study
            params_dyn = study_config[method_name]['params']['dynamic']

            study_results[method_name] = dict()

            for param_name_dyn in params_dyn.keys():
                print(f'    * varying {param_name_dyn}')

                param_range_dyn = params_dyn[param_name_dyn]

                # fixed parameters without parameter `param_name` 
                params_fix_kw = {
                    k: v for (k,v) in params_fix.items() if k != param_name_dyn
                }

                study_results[method_name][param_name_dyn] = dict()

                for test_func in test_functions:
                    print(f'      * @ {test_func.name}', end='\t\t', flush=True)

                    study_results[method_name][param_name_dyn][test_func.name] = dict()
                    study_results[method_name][param_name_dyn][test_func.name] = dict()

                    _histories = list()

                    for param_dyn in param_range_dyn:

                        params_dyn_kw = {
                            param_name_dyn : param_dyn
                        }

                        for seed in seeds:
                            optimizer = opt(
                                test_func.objective, test_func.area,
                                seed=seed, **params_fix_kw, **params_dyn_kw
                            )
                            history = optimizer.optimize()
                            # remove points because dict gets really big
                            history.pop('points')
                            _histories.append(history)

                    study_results[method_name][param_name_dyn][test_func.name]['histories'] = np.squeeze(
                        np.array(_histories).reshape(-1, n_retries_sensitivity)
                    )

                    print('DONE!')

        print('* end sensitivity study')

    #
    # PLOT SENSITIVITY STUDY
    #

    if SENSITIVITY_STUDY_PLOT:
        for method_name in study_config.keys():
            params = study_config[method_name]['params']

            for param_name, label in zip(params['dynamic'].keys(), params['labels']):

                sapso.plot.sensitivity_study_plot(
                    test_functions, study_config, study_results,
                    method=method_name, param=param_name, xlabel=label,
                    layout=layout, figsize=figsize,
                    save=True, path=f'hyperparam_search__{method_name}_{param_name}.pdf'
                )

    #
    # BENCHMARK FUNCTION VISUALIZATION
    #

    if BENCHMARK_FUNCTION_CONTOUR_PLOT:
        sapso.plot.testfunction_contour_plot(
            benchmark_functions, layout, figsize, show=False, save=True,
            path='overview_benchmarkfunction_contour.pdf'
        )

    if BENCHMARK_FUNCTION_SURFACE_PLOT:
        sapso.plot.testfunction_surface_plot(
            benchmark_functions, z_offsets_benchmark, layout, figsize,
            show=False, save=True,
            path='overview_benchmarkfunction_surface.pdf'
        )

    #
    # BENCHMARK
    #

    if BENCHMARK:

        benchmark_config = {
            'SA' : {
                'implementation' : sapso.sa.SimulatedAnnealing,
                'params' : {
                    'iterations'  : 80000,
                    'step_size'   : 0.2,
                    'temperature' : sapso.sa.temperature_exp,
                }
            },
            'PSO' : {
                'implementation' : sapso.pso.ParticleSwarmOptimization,
                'params' : {
                    'iterations'  : 400,
                    'n_particles' : 200,
                    'w'           : 0.7,
                    'a_ind'       : 1.0,
                    'a_neigh'     : 1.6,
                }
            },
            'APSO' : {
                'implementation' : sapso.pso.AdaptiveParticleSwarmOptimization,
                'params' : {
                    'iterations'    : 500,
                    'n_particles'   : 200,
                    'w'             : 0.75,
                    'a_ind'         : 1.0,
                    'a_neigh'       : 2.0,
                    'interpolation' : 'const',
                }
            }
        }

        n_methods = len(benchmark_config.keys())
        n_seeds = n_methods * n_retries_benchmark

        assert max_seed >= n_seeds, f'Seed pool too small. {max_seed} < {n_seeds}'

        master_rng = np.random.default_rng(seed=master_seed)
        seed_pool = master_rng.choice(
            max_seed, size=n_seeds, replace=False
        ).reshape(n_methods, n_retries_benchmark)

        # holding all the results
        benchmark_results = dict()

        print('* begin benchmark')

        for method_name, seeds in zip(benchmark_config.keys(), seed_pool):
            print(f'  * {method_name}')

            fncall_wrapper = sapso.wrappers.FunctionCallCounterWrapper
            # optimization method, i.e. SA, PSO, APSO
            opt = benchmark_config[method_name]['implementation']
            params = benchmark_config[method_name]['params']

            benchmark_results[method_name] = dict()

            for bench_func in benchmark_functions:
                print(f'    * @ {bench_func.name}', end='\t\t', flush=True)

                benchmark_results[method_name][bench_func.name] = dict()

                _histories = list()

                for seed in seeds:
                    optimizer = fncall_wrapper(
                        opt, bench_func.objective, bench_func.area,
                        seed=seed, **params
                    )

                    history = optimizer.optimize()
                    # remove points because dict gets really big
                    history.pop('points')
                    _histories.append(history)

                benchmark_results[method_name][bench_func.name]['histories'] = np.squeeze(
                    np.array(_histories).reshape(-1, n_retries_benchmark)
                )

                print('DONE!')

        print('* end benchmark')

    #
    # PLOT BENCHMARK
    #

    if BENCHMARK_PLOT:

        print('* begin benchmark plot')
        for method_name in benchmark_config.keys():
            print(f'  * {method_name}')

            sapso.plot.benchmark_plot(
                benchmark_functions, benchmark_config, benchmark_results,
                method=method_name, layout=layout, figsize=figsize,
                save=True, path=f'benchmark__{method_name}.pdf'
            )
        print('* end benchmark plot')


if __name__ == '__main__':
    if os.getcwd().split('/')[-1] != 'report':
        exit('Please run as:\n  python3 generate_plots.py')

    main()
