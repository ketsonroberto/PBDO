from numpy.random import randint
from random import random as rnd
from random import gauss, randrange
import scipy as sp
import numpy as np
import itertools
import copy

from StochasticMechanics import Stochastic
from BuildingProperties import *
import Performance

class ASGD():
    '''
    ASGD main class

    fun: objective function to be minimized (callable)
    grad: gradient of theobjective function to be minimized (callable)
    sampler: generates samples of the random variables. Takes integer n as input
        and returns n samples (callable)
    x0: starting point of optimizaton (array)
    iters: number of iterations when calling run method (int)
    step: step-size to use (float)
    ascent: flag to maximize instead of minimize (boolean)
    L and U: lower and upper bounds for the design parameters (arrays)
    gtol: tolerance for the norm of the gradient used as a stop criteria (float)
    print: flag for printing data on screen
    '''
    def __init__(self,
                 fun,
                 grad,
                 sampler,
                 x0,
                 iters=1000,
                 step=1e-3,
                 smooth=0.9999,
                 ascent=False,
                 L=[],
                 U=[],
                 gtol=0,
                 print=True):
        self.fun = fun
        self.grad = grad
        self.x = [np.array(x0)]
        self.x_ = np.zeros(np.shape(x0))
        self.x_pr = [np.array(x0)]
        self.grads = []
        self.sampler = sampler
        self.step = step
        self.iters = iters
        self.ascent = 1 - 2*ascent
        self.smooth = smooth
        self.print = print
        self.k = 1
        self.gamma = .0
        self.lamb = 1.
        self.q = .0
        self.L = L
        self.U = U
        self.gtol = gtol
        self.z_ = np.zeros(np.shape(x0))

    def __call__(self):
        sample = self.sampler(1)
        self.grads.append(self.grad(self.x[-1], sample[0]))
        z = self.x[-1] - self.step/(self.k)**.5*self.grads[-1] * self.ascent
        if ((z + self.gamma*(z-self.z_) - self.x[-1]) @
                self.grads[-1]*self.ascent) > 0:
            self.lamb = 1.0
            self.gamma = 0.0
        x_ = bounds(z + self.gamma*(z-self.z_), self.L, self.U)
        self.x.append(x_)
        self.z_ = np.array(z)
        lamb = (self.q-self.lamb**2)/2 \
            + np.sqrt(self.q**2 - 2*self.q*self.lamb**2
                      + self.lamb**4 + 4*self.lamb**2)/2
        self.gamma = self.lamb*(1-self.lamb)/(self.lamb**2+lamb)
        self.lamb = np.array(lamb)
        self.k += 1
        self.pr_update()
        if self.print: print(f'x:{self.x[-1]}, averaged x: {self.x_pr[-1]}'
                             f'grad norm:{np.linalg.norm(self.grads[-1])}')
        return

    def run(self):
        k = 0
        g_norm = self.gtol*2
        while g_norm >= self.gtol and (k < self.iters):
            k += 1
            self()
            g_norm = np.linalg.norm(self.grads[-1])

    def pr_update(self):
        '''Polyak--Ruppert averaging'''
        # weight = 1/(self.k/2)**.5
        # weight_ = np.sum([1/(k_)**.5 for k_ in range(1,int(self.k/2 - 1))])
        # weights = [1/(k_)**.5 for k_ in range(1,int(self.k)+1)]
        # x_ = [x*w for x,w in zip(self.x, weights)]
        # x_pr = np.sum(x_, axis=0)/np.sum(weights)
        beta = self.smooth
        self.x_ = self.x_*beta + self.x[-1]*(1-beta)
        self.x_pr.append(self.x_/(1-beta**self.k))

    @staticmethod
    def bounds(x, L, U):
        x = np.maximum(x, L)
        x = np.minimum(x, U)
        return x


########################################################################################################################
########################################################################################################################
#                                            Heuristic                                                                 #
########################################################################################################################
########################################################################################################################

class GeneticAlgorithm:

    def __init__(self, opt_object=None, args=None):
        self.opt_object = opt_object
        self.args = args

    def individual(self, number_of_genes, upper_limit, lower_limit):
        individual = [round(rnd() * (upper_limit - lower_limit) + lower_limit, 1) for x in range(number_of_genes)]

        return individual

    def population(self, number_of_individuals, number_of_genes, upper_limit, lower_limit):
        return [self.individual(number_of_genes, upper_limit, lower_limit) for x in range(number_of_individuals)]

    def fitness_calculation(self, indiv):
        ndof = building["ndof"]
        size_col = []
        for i in range(ndof):
            size_col.append(indiv[i])

        size_col = np.array(size_col)
        fitness_value = self.opt_object.objective_function(size_col=size_col, args=self.args)

        return fitness_value

    def roulette(self,cum_sum, chance):
        veriable = list(cum_sum.copy())
        veriable.append(chance)
        veriable = sorted(veriable)
        return veriable.index(chance)

    def selection(self,generation, method='Fittest Half'):
        generation['Normalized Fitness'] = \
            sorted([generation['Fitness'][x] / sum(generation['Fitness'])
                    for x in range(len(generation['Fitness']))], reverse=True)
        generation['Cumulative Sum'] = np.array(
            generation['Normalized Fitness']).cumsum()
        if method == 'Roulette Wheel':
            selected = []
            for x in range(len(generation['Individuals']) // 2):
                selected.append(self.roulette(generation
                                         ['Cumulative Sum'], rnd()))
                while len(set(selected)) != len(selected):
                    selected[x] = \
                        (self.roulette(generation['Cumulative Sum'], rnd()))
            selected = {'Individuals':
                            [generation['Individuals'][int(selected[x])]
                             for x in range(len(generation['Individuals']) // 2)]
                , 'Fitness': [generation['Fitness'][int(selected[x])]
                              for x in range(
                        len(generation['Individuals']) // 2)]}
        elif method == 'Fittest Half':
            selected_individuals = [generation['Individuals'][-x - 1]
                                    for x in range(int(len(generation['Individuals']) // 2))]
            selected_fitnesses = [generation['Fitness'][-x - 1]
                                  for x in range(int(len(generation['Individuals']) // 2))]
            selected = {'Individuals': selected_individuals,
                        'Fitness': selected_fitnesses}
        elif method == 'Random':
            selected_individuals = \
                [generation['Individuals']
                 [randint(1, len(generation['Fitness']))]
                 for x in range(int(len(generation['Individuals']) // 2))]
            selected_fitnesses = [generation['Fitness'][-x - 1]
                                  for x in range(int(len(generation['Individuals']) // 2))]
            selected = {'Individuals': selected_individuals,
                        'Fitness': selected_fitnesses}
        return selected

    def pairing(self, elit, selected, method='Fittest'):
        individuals = [elit['Individuals']] + selected['Individuals']
        fitness = [elit['Fitness']] + selected['Fitness']
        if method == 'Fittest':
            parents = [[individuals[x], individuals[x + 1]]
                       for x in range(len(individuals) // 2)]
        if method == 'Random':
            parents = []
            for x in range(len(individuals) // 2):
                parents.append(
                    [individuals[randint(0, (len(individuals) - 1))],
                     individuals[randint(0, (len(individuals) - 1))]])
                while parents[x][0] == parents[x][1]:
                    parents[x][1] = individuals[
                        randint(0, (len(individuals) - 1))]
        if method == 'Weighted Random':
            normalized_fitness = sorted(
                [fitness[x] / sum(fitness)
                 for x in range(len(individuals) // 2)], reverse=True)
            cummulitive_sum = np.array(normalized_fitness).cumsum()
            parents = []
            for x in range(len(individuals) // 2):
                parents.append(
                    [individuals[self.roulette(cummulitive_sum, rnd())],
                     individuals[self.roulette(cummulitive_sum, rnd())]])
                while parents[x][0] == parents[x][1]:
                    parents[x][1] = individuals[
                        self.roulette(cummulitive_sum, rnd())]
        return parents

    def mating(self, parents, method='Single Point'):
        if method == 'Single Point':
            pivot_point = randint(1, len(parents[0]))
            offsprings = [parents[0] \
                              [0:pivot_point] + parents[1][pivot_point:]]
            offsprings.append(parents[1]
                              [0:pivot_point] + parents[0][pivot_point:])
        if method == 'Two Pionts':
            pivot_point_1 = randint(1, len(parents[0] - 1))
            pivot_point_2 = randint(1, len(parents[0]))
            while pivot_point_2 < pivot_point_1:
                pivot_point_2 = randint(1, len(parents[0]))
            offsprings = [parents[0][0:pivot_point_1] +
                          parents[1][pivot_point_1:pivot_point_2] +
                          [parents[0][pivot_point_2:]]]
            offsprings.append([parents[1][0:pivot_point_1] +
                               parents[0][pivot_point_1:pivot_point_2] +
                               [parents[1][pivot_point_2:]]])
        return offsprings

    def mutation(self, individual, upper_limit, lower_limit, muatation_rate=2,
                 method='Reset', standard_deviation=0.001):
        gene = [randint(0, 7)]
        for x in range(muatation_rate - 1):
            gene.append(randint(0, 7))
            while len(set(gene)) < len(gene):
                gene[x] = randint(0, 7)
        mutated_individual = individual.copy()
        if method == 'Gauss':
            for x in range(muatation_rate):
                mutated_individual[x] = \
                    round(individual[x] + gauss(0, standard_deviation), 1)
        if method == 'Reset':
            for x in range(muatation_rate):
                mutated_individual[x] = round(rnd() * \
                                              (upper_limit - lower_limit) + lower_limit, 1)
        return mutated_individual

    def first_generation(self, pop):
        fitness = [self.fitness_calculation(pop[x]) for x in range(len(pop))]
        sorted_fitness = sorted([[pop[x], fitness[x]] for x in range(len(pop))], key=lambda x: x[1])
        population = [sorted_fitness[x][0] for x in range(len(sorted_fitness))]
        fitness = [sorted_fitness[x][1] for x in range(len(sorted_fitness))]
        return {'Individuals': population, 'Fitness': sorted(fitness)}

    def next_generation(self, gen, upper_limit, lower_limit):
        elit = {}
        next_gen = {}
        elit['Individuals'] = gen['Individuals'].pop(-1)
        elit['Fitness'] = gen['Fitness'].pop(-1)
        selected = self.selection(gen)
        parents = self.pairing(elit, selected)
        offsprings = [[[self.mating(parents[x])
                        for x in range(len(parents))]
                       [y][z] for z in range(2)]
                      for y in range(len(parents))]
        offsprings1 = [offsprings[x][0]
                       for x in range(len(parents))]
        offsprings2 = [offsprings[x][1]
                       for x in range(len(parents))]
        unmutated = selected['Individuals'] + offsprings1 + offsprings2
        mutated = [self.mutation(unmutated[x], upper_limit, lower_limit)
                   for x in range(len(gen['Individuals']))]
        unsorted_individuals = mutated + [elit['Individuals']]
        unsorted_next_gen = \
            [self.fitness_calculation(mutated[x]) for x in range(len(mutated))]
        unsorted_fitness = [unsorted_next_gen[x] for x in range(len(gen['Fitness']))] + [elit['Fitness']]
        sorted_next_gen = \
            sorted([[unsorted_individuals[x], unsorted_fitness[x]] for x in range(len(unsorted_individuals))],
                   key=lambda x: x[1])
        next_gen['Individuals'] = [sorted_next_gen[x][0]
                                   for x in range(len(sorted_next_gen))]
        next_gen['Fitness'] = [sorted_next_gen[x][1]
                               for x in range(len(sorted_next_gen))]
        gen['Individuals'].append(elit['Individuals'])
        gen['Fitness'].append(elit['Fitness'])
        return next_gen

    def fitness_similarity_chech(self,max_fitness, number_of_similarity):
        result = False
        similarity = 0
        for n in range(len(max_fitness) - 1):
            if max_fitness[n] == max_fitness[n + 1]:
                similarity += 1
            else:
                similarity = 0
        if similarity == number_of_similarity - 1:
            result = True
        return result


