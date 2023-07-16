import itertools
import numpy as np

from . import utils
from . import base
from . import sa

__all__ = ['Particle', 'ParticleSwarmOptimization', 'AdaptiveParticleSwarmOptimization']

class Particle:
    # TODO(LOW): add Particle docstring

    def __init__(self, owner, objective, area, w, a_ind, a_neigh):
        # owner of particle, self.id is relative to owner
        self.owner = owner
        # unique particle ID
        self.id = self.owner._new_id()
        # objective / cost / fitness function
        self.objective = objective
        # bounding hyper cube of search space
        self.area = utils.validate_area(area)
        # dimensionality of optimization problem
        self.n = self.area.shape[0]
        # inertia weight, typically in [0, 1)
        self.w = w
        # cognitive coefficient (i.e. individual behaviour), typically in [1, 3]
        self.a_ind = a_ind
        # social coefficient (i.e. group behaviour), typically in [1, 3]
        self.a_neigh = a_neigh

        self.pos = utils.uniform(self.owner.rng, area)
        self.val = self.objective(*self.pos)

        # individual best position and corresponding value
        self.pos_best_ind = np.array(self.pos)
        self.val_best_ind = self.objective(*self.pos_best_ind)

        # group best position and corresponding value
        self.pos_best_neigh = np.array(self.pos)
        self.val_best_neigh = self.objective(*self.pos_best_neigh)

        # velocity of particle in R^n
        # initialization according to wikipedia
        # https://en.wikipedia.org/wiki/Particle_swarm_optimization#Algorithm
        self.vel = utils.uniform(self.owner.rng, np.hstack([-np.diff(area), np.diff(area)]))

    def update(self):
        r_ind = self.owner.rng.random(self.n)
        r_neigh = self.owner.rng.random(self.n)

        dir_best_ind = utils.normalize(self.pos_best_ind - self.pos)
        dir_best_neigh = utils.normalize(self.pos_best_neigh - self.pos)

        self.vel = (
            self.w * self.vel
            + self.a_ind * r_ind * dir_best_ind
            + self.a_neigh * r_neigh * dir_best_neigh
        )

        self.pos = np.clip(self.pos + self.vel, *self.area.T)

    def eval(self):
        self.val = self.objective(*self.pos)

        if np.less(self.val, self.val_best_ind):
            self.pos_best_ind = np.copy(self.pos)
            self.val_best_ind = np.copy(self.val)

        if np.less(self.val, self.val_best_neigh):
            self.pos_best_neigh = np.copy(self.pos)
            self.val_best_neigh = np.copy(self.val)


class ParticleSwarmOptimization(base.OptimizationMethod):
    # TODO(LOW): add PSO docstring
    def __init__(self,
                 objective, area,
                 iterations=400,
                 seed=42,
                 n_particles=200,
                 w=0.7,            # inertial decay,        in [0,1)
                 a_ind=1.0,        # cognitive coefficient, in [1,3]
                 a_neigh=1.6,      # social coefficient,    in [1,3]
                 goal='min'        # optimization goal,     in ['min', 'max']
                 ):
        super().__init__(objective, area, iterations, seed, goal)
        # counter for generating increasing Particle IDs (relative to self)
        self._new_id = itertools.count().__next__
        # PSO-specific parameters
        self.params = base.Params()
        # number of particles
        self.params.n_particles = n_particles
        # inertia weight
        self.params.w = w
        # attraction towards individual best
        self.params.a_ind = a_ind
        # attraction towards neighbour best
        self.params.a_neigh = a_neigh

        self.particles = [
            Particle(self, self.objective, self.area, self.params.w, self.params.a_ind, self.params.a_neigh)
            for _ in range(n_particles)
        ]

    def _reset_history(self):
        # track history of encountered points
        self.history = {'points': [], 'values': [], 'particle_id': []}

    def _update_history(self, particle):
        """
        CAN ONLY BE CALLED AFTER _reset_history
        CAN ONLY BE CALLED BEFORE _finalize_history
        """
        # keep track of point and value
        self.history['points'].append(particle.pos)
        self.history['values'].append(particle.val)
        self.history['particle_id'].append(particle.id)

    def _finalize_history(self):
        """
        CAN ONLY BE CALLED AFTER _reset_history
        """

        # ensure the history uses numpy arrays, facilitates plotting with matplotlib
        self.history = {
            # results
            'points'      : np.array(self.history['points']),
            'values'      : np.array(self.history['values']),
            'particle_id' : np.array(self.history['particle_id']),
            'best_val'    : self.best_val,
            'best_point'  : self.best_pos,
            # meta information
            'meta'        : {
                'method'       : 'PSO',
                'params'       : {
                    'iterations'  : self.iterations,
                    'seed'        : self.seed,
                    'goal'        : self.goal,
                    'n_particles' : self.params.n_particles, 
                    'w'           : self.params.w,
                    'a_ind'       : self.params.a_ind,
                    'a_neigh'     : self.params.a_neigh
                },
            },
        }


    def optimize(self):
        self._reset_history()

        init_val = np.array([p.val for p in self.particles])
        init_pos = np.array([p.pos for p in self.particles])

        self.best_val = np.min(init_val)
        self.best_pos = init_pos[np.argmin(init_val)]

        for _ in range(self.iterations):
            for particle in self.particles:

                particle.val_best_neigh = self.best_val
                particle.pos_best_neigh = self.best_pos

                particle.update()
                particle.eval()

                self._update_history(particle)

                self.best_val = particle.val_best_neigh
                self.best_pos = particle.pos_best_neigh

        self._finalize_history()

        return self.history

class AdaptiveParticle(Particle):
    # TODO(LOW): add Particle docstring

    def __init__(self, owner, objective, area, w, a_ind, a_neigh, weights):
        super().__init__(owner, objective, area, w, a_ind, a_neigh)
        # component weights
        self.weights = np.array(weights)

    def update(self, iteration):
        r_ind = self.owner.rng.random(self.n)
        r_neigh = self.owner.rng.random(self.n)

        dir_best_ind = utils.normalize(self.pos_best_ind - self.pos)
        dir_best_neigh = utils.normalize(self.pos_best_neigh - self.pos)

        # leftoff
        self.vel = (
            self.weights[0, iteration] * self.w * self.vel
            + self.weights[1, iteration] * self.a_ind * r_ind * dir_best_ind
            + self.weights[2, iteration] * self.a_neigh * r_neigh * dir_best_neigh
        )

        self.pos = np.clip(self.pos + self.vel, *self.area.T)

class AdaptiveParticleSwarmOptimization(base.OptimizationMethod):
    # TODO(LOW): add PSO docstring
    def __init__(self,
                 objective, area,
                 iterations=500,
                 seed=42,
                 n_particles=200,
                 components=None,        # weighting of velocity components
                 interpolation='const',  # interpolation method   in ['cubic', 'exp', 'const']
                 w=0.75,                 # inertial decay,        in [0,1)
                 a_ind=1.0,              # cognitive coefficient, in [1,3]
                 a_neigh=2.0,            # social coefficient,    in [1,3]
                 goal='min'              # optimization goal,     in ['min', 'max']
                 ):
        super().__init__(objective, area, iterations, seed, goal)
        # counter for generating increasing Particle IDs (relative to self)
        self._new_id = itertools.count().__next__
        # 
        if components is None:
            components = [
                # component 0 - inertial component
                np.array([
                    (0.00, 1.00),
                    (0.33, 0.33),
                    (0.67, 0.33),
                    (1.00, 0.33)
                ]),
                # component 1 - individual best component
                np.array([
                    (0.00, 0.00),
                    (0.33, 0.67),
                    (0.67, 0.33),
                    (1.00, 0.33)
                ]),
                # component 2 - neighbour best component
                np.array([
                    (0.00, 0.00),
                    (0.33, 0.00),
                    (0.67, 0.33),
                    (1.00, 0.33)
                ])
            ]

            if interpolation == 'cubic':

                def interpolate_cubic(points):
                    coeffs = np.polyfit(*points.T, deg=3)
                    xs = np.linspace(0, 1, self.iterations)
                    ys = np.sum([ 
                        c * xs ** (3-i) for i,c in enumerate(coeffs) 
                    ], axis=0)
                    return ys

                interpolate = interpolate_cubic

            elif interpolation == 'exp':

                def interpolate_exp(points):
                    log_points = np.vstack([
                            points[:, 0], np.log(np.maximum(points[:, 1], 2e-3))
                    ]).T

                    xs = np.linspace(0, 1, self.iterations)
                    ys = list()

                    for i in range(3):
                        _points = log_points[i:i+2]
                        coeffs = np.polyfit(*_points.T, deg=1)
                        lo, hi = _points[:, 0]
                        x = xs[np.where((lo <= xs) & (xs <= hi))]
                        y = np.exp(coeffs[0] * x + coeffs[1])
                        ys.append(y)

                    ys = np.concatenate(ys)
                    return ys

                interpolate = interpolate_exp

            elif interpolation == 'const':

                def interpolate_const(points):
                    xs = np.linspace(0, 1, self.iterations)
                    ys = list()

                    for i in range(3):
                        _points = points[i:i+2]
                        coeffs = np.polyfit(*_points.T, deg=0)
                        lo, hi = _points[:, 0]
                        x = xs[np.where((lo <= xs) & (xs <= hi))]
                        y = np.ones_like(x) * coeffs[0]
                        ys.append(y)

                    ys = np.concatenate(ys)
                    return ys

                interpolate = interpolate_const

            else:
                raise ValueError(
                    f'interpolation method {interpolation} not recognized!'
                )

            # interpolate between points
            weights = np.array([ interpolate(c) for c in components ])

            # ensure compoents sum to 1.0
            weights = weights / np.sum(weights, axis=0)
        else:
            raise NotImplementedError(
                f'sorry no time for implementation'
            )

        # Adaptive PSO-specific parameters
        self.params = base.Params()

        # number of particles
        self.params.n_particles = n_particles
        # inertia weight
        self.params.w = w
        # attraction towards individual best
        self.params.a_ind = a_ind
        # attraction towards neighbour best
        self.params.a_neigh = a_neigh
        # component weights
        self.params.weights = weights

        self.particles = [
            AdaptiveParticle(self, self.objective, self.area, self.params.w, self.params.a_ind, self.params.a_neigh, self.params.weights)
            for _ in range(n_particles)
        ]

    def _reset_history(self):
        # track history of encountered points
        self.history = {'points': [], 'values': [], 'particle_id': []}

    def _update_history(self, particle):
        """
        CAN ONLY BE CALLED AFTER _reset_history
        CAN ONLY BE CALLED BEFORE _finalize_history
        """
        # keep track of point and value
        self.history['points'].append(particle.pos)
        self.history['values'].append(particle.val)
        self.history['particle_id'].append(particle.id)

    def _finalize_history(self):
        """
        CAN ONLY BE CALLED AFTER _reset_history
        """

        # ensure the history uses numpy arrays, facilitates plotting with matplotlib
        self.history = {
            # results
            'points'      : np.array(self.history['points']),
            'values'      : np.array(self.history['values']),
            'particle_id' : np.array(self.history['particle_id']),
            'best_val'    : self.best_val,
            'best_point'  : self.best_pos,
            # meta information
            'meta'        : {
                'method'       : 'APSO',
                'params'       : {
                    'iterations'  : self.iterations,
                    'seed'        : self.seed,
                    'goal'        : self.goal,
                    'n_particles' : self.params.n_particles,
                    'w'           : self.params.w,
                    'a_ind'       : self.params.a_ind,
                    'a_neigh'     : self.params.a_neigh,
                    'weights'     : self.params.weights
                },
            },
        }


    def optimize(self):
        self._reset_history()

        init_val = np.array([p.val for p in self.particles])
        init_pos = np.array([p.pos for p in self.particles])

        self.best_val = np.min(init_val)
        self.best_pos = init_pos[np.argmin(init_val)]

        for iteration in range(self.iterations):
            for particle in self.particles:

                particle.val_best_neigh = self.best_val
                particle.pos_best_neigh = self.best_pos

                particle.update(iteration)
                particle.eval()

                self._update_history(particle)

                self.best_val = particle.val_best_neigh
                self.best_pos = particle.pos_best_neigh

        self._finalize_history()

        return self.history
