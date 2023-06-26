# Optimization-Methods

Optimization Methods for Engineers Project (Spring Semester 23)

## TODOs

* [x] SA implementation - debug with plots
* [x] PSO implementation - debug with plots
- [x] PSO code clean up
- [ ] clean up TODOs in code
* [ ] Combination SA PSO implementation - debug with plots
* [ ] search suitable testing functions
* [x] implement plot comparing different optimization methods
* [x] stylize plots coherently for report
* [ ] write report

## Ideas for Report

- run experiments multiple times with different seeds
    - calculate average difference between 
        - found optimum and global optimum (position and value)
    - this acts as certificate for the algorithm's robustness
    - potentially high number of agents
    - this gives statistical PSO plot 

- cherry pick nice plots, which visually describe how each optimization method works
    - e.g. for PSO: plots with only few number of agents

- improvement of PSO by
    - normalizing direction (individual, group optima)
    - making (individual, group) acceleration dependent on (area | search space)
    - introduce randomness through rotating the (individual, group) direction vector

- disadvantages of SA
    - depends on initial position

- disadvantages of PSO
    - need high number of agents (m) in order to explore search space thoroughly enough
    - many tunable hyper-parameters

- bringing agents into SA:
    - have multiple agents perform SA simultaneously (on its own)
        - is equivalent to running SA multiple times

- improve visual quality of plots 
    - nicely style plot title (i.e. with parameters chosen)
    - make plots with both kinds of histories
        - one where keeps track of _all_ points
        - one where keeps track of _only best_ points
    - PSO:
        - by encoding history via alpha colour channel
            - i.e. old points are strongly opaque, whereas new points are weakly opaque
            - maybe connect points by arrows / lines
    - SA:
        - similarly to PSO, bring history of temperature in via alpha channel

- explicitly conform to report structure as specified by professor!
- forge report with *quality over quantity*
    - why did we do stuff 
    - nice concise plot legends
    - simple + few sentences

- statistical plots
    - time measurements (how long optimization methods take)
    - what percentage of the search space do optimization methods explore

- show strengths & weaknesses / limitations
    - SA
    - PSO
    - SAPSO

- don't explain inner workings of SA & PSO
    - professor knows how they work
    - except for additional improvements 
        - such as added randomness in PSO (individual, group) direction

## Ideas for Combination of SA x PSO

- instead of clipping points outside of search space, make them get reflected at borders of the search space

- velocity update is split into
    - self component
    - individual optimum component
    - group optimum component

- velocity update rule transition depends on Boltzmann function of temperature, 
    - such that its a continuous transition rather than abrupt
    - scaling factors of importance of each component is dependent of Boltzmann function
        - make each component depends on Boltzmann function 
            - all factors sum to one
            - component1 : 1.0 .... 0.3 .... 0.3
            - component2 : 0.0 .... 0.7 .... 0.3
            - component3 : 0.0 .... 0.0 .... 0.3
            - where .... is smooth by Boltzmann and temperature
                - temperature is dependent on current / max iterations

- phase 0
    - agents explore initial (fixed) direction all for themselves
    - velocity update only depends on self component

- phase 1
    - agents explore their current optimum found along approximately initial direction
    - velocity update only depends on self component and individual optimum component

- phase 2 
    - all agents explore the currently observed group optimum together 
    - velocity update now depends on all three components

## Testing Functions

For the following cases, (at least) one testing functions is needed.
This allows us to conclude about the **strengths** as well as the **weaknesses** of each method.

- SA is good / bad
- PSO is good / bad
- SAPSO is good / bad

## Resources

- [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
- [Particle Swarm Optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization)
- [Testing Functions](https://en.wikipedia.org/wiki/Test_functions_for_optimization)


