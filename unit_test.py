import sapso

tfunc = sapso.test.Eggholder()

kwargs_sa = { 'iterations' : 100, 'seed' : 42, 'temperature' : None, 'step_size' : 0.1, 'goal' : 'min' }
kwargs_pso = { 'iterations' : 100, 'seed' : 42, 'n_particles' : 10, 'w' : 0.75, 'a_ind' : 1., 'a_neigh' : 2., 'goal' : 'min' }

SA = sapso.sa.SimulatedAnnealing(tfunc.objective, tfunc.area, **kwargs_sa)
PSO = sapso.pso.ParticleSwarmOptimization(tfunc.objective, tfunc.area, **kwargs_pso)

history_sa = SA.optimize()
history_pso = PSO.optimize()
