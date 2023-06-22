import itertools
import numpy as np

from . import utils


class Particle:
    # TODO: add docstring with input, methods and properties

    # (user hidden) function to keep track of unique incremental IDs (thread-safe?)
    __new_id = itertools.count().__next__

    def __init__(self, objective, area, a_ind, a_neigh, w, goal='min'):
        # unique particle ID
        self.id = self.__class__.__new_id()
        # objective / cost / fitness function
        self.objective = objective
        # bounding hyper cube of search space
        self.area = utils.validate_area(area)
        # dimensionality of optimization problem
        self.n = self.area.shape[0]
        # goal of optimization (minimization / maximization)
        self.goal = utils.validate_goal(goal)
        self.better, _, _ = utils.comparison_funcs_from_goal(self.goal)
        # inertia weight, typically in [0, 1)
        self.w = w
        # cognitive coefficient (i.e. individual behaviour), typically in [1, 3]
        self.a_ind = a_ind
        # social coefficient (i.e. group behaviour), typically in [1, 3]
        self.a_neigh = a_neigh

        # TODO: could be optimized but needs attention to not coupple variables together!

        # position and value of particle in $R^n$ and $R$
        self.pos = utils.uniform(area)
        self.val = self.objective(self.pos)

        # individual best position and corresponding value
        self.pos_best_ind = np.array(self.pos)
        self.val_best_ind = self.objective(self.pos_best_ind)

        # group best position and corresponding value
        self.pos_best_neigh = np.array(self.pos)
        self.val_best_neigh = self.objective(self.pos_best_neigh)

        # velocity of particle in R^n
        # initialization according to wikipedia
        # https://en.wikipedia.org/wiki/Particle_swarm_optimization#Algorithm
        self.vel = utils.uniform(np.hstack(-np.diff(area), np.diff(area)))

    def update(self):
        r_ind = np.random.rand(self.n)
        r_neigh = np.random.rand(self.n)

        dir_best_ind = utils.normalize(self.pos_best_ind - self.pos)
        dir_best_neigh = utils.normalize(self.pos_best_neigh - self.pos)

        self.vel = (
            self.w * self.vel
            + self.a_ind * r_ind * dir_best_ind
            + self.a_neigh * r_neigh * dir_best_neigh
        )

        self.pos = np.clip(self.pos + self.vel, *self.area.T)

    def eval(self):
        self.val = self.objective(self.pos)

        if better(self.val, self.val_best_ind):
            self.pos_best_ind = np.copy(self.pos)
            self.val_best_ind = np.copy(self.val)

        if better(self.val, self.val_best_neigh):
            self.pos_best_neigh = np.copy(self.pos)
            self.val_best_neigh = np.copy(self.val)

def particle_swarm_optimization(objective, area, n_particles, iterations):
    # TODO bring into params

    seed = 7

    # params according to wikipedia
    w = 0.75      # inertial decay,        in [0,1)
    a_ind   = 1   # cognitive coefficient, in [1,3]
    a_neigh = 2   # social coefficient,    in [1,3]
    goal = 'min'  # optimization goal,     in ['min', 'max']

    goal = utils.validate_goal(goal)
    _, extremum, argextremum = utils.comparison_funcs_from_goal(goal)

    particles = [
        Particle(objective, area, a_ind, a_neigh, w, goal)
        for _ in range(n_particles)
    ]

    init_val = np.array([p.val for p in particles])
    init_pos = np.array([p.pos for p in particles])

    best_val = extremum(init_val)
    best_pos = init_pos[argextremum(init_val)]

    # track history of encountered points
    history = {'points': [], 'values': [], 'particle_id': []}

    for _ in range(iterations):
        for particle in particles:

            particle.val_best_neigh = best_val
            particle.pos_best_neigh = best_pos

            particle.update()
            particle.eval()

            # keep track of point and value
            history['points'].append(particle.pos)
            history['values'].append(particle.val)
            history['particle_id'].append(particle.id)

            best_val = particle.val_best_neigh
            best_pos = particle.pos_best_neigh

    # ensure the history uses numpy arrays, facilitates plotting with matplotlib
    history = {
        # results
        'points'      : np.array(history['points']),
        'values'      : np.array(history['values']),
        'particle_id' : np.array(history['particle_id']),
        'best_point'  : best_pos,
        'best_val'    : best_val,
        # meta information
        'algorithm'   : 'particle_swarm',
        'params'      : {
            'goal'       : goal,
            'seed'       : seed,
            'iterations' : iterations,
            'w'          : w,
            'a_ind'      : a_ind,
            'a_neigh'    : a_neigh
        },
    }

    return history

