# SAPSO - a tiny optimization library in python

The idea for this library arose when for the course Optimization Methods for Engineers at ETH Zurich I had to implement the optimization algorithms [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing) and [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
These algorithms, for brevity dubbed `sa` and `pso` respectively, hence gave rise to library's name by concatenating the two abbreviations together.

For the course project my friend and I decided to do a quantitative study which tests, benchmarks and ultimately compares Simulated Annealing, Particle Swarm Optimization and a novel algorithm that was devised based on the strengths and weaknesses of the former two.
Thus one will find additional code for various different 2D test and benchmark functions, which were used for the experiments conducted in our study, as well as plotting routines that try to visualize the characteristics of these experiments in an aesthetic way.

For the full final report of the course, which was compiled in the style of a presentation, see [here](https://github.com/thmeier/Optimization-Methods/blob/main/report/presentation/report.pdf).

In the future, once I find the time, I would like to implement the remaining algorithms introduced in the previously mentioned course and use them in some of my personal projects.
