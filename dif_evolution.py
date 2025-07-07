import networkx as nx
import numpy as np
import random
from unit import getNewSolutionPriori,newG, getCommunity
from myevaluation import get_eva

class DifferentialEvolution():
    def __init__(self, G, detectionFun, target_community, num_attack, distribute):
        self.G = G
        self.detectionFun = detectionFun
        self.target_community = target_community
        self.num_attack = num_attack
        self.distribute = distribute


    def fitness_func(self, individual):
        newGraph = newG(individual, self.G)  # 根据新生成的解对图进行处理
        newcom = getCommunity(newGraph, self.detectionFun)
        eva_score = get_eva(newGraph, self.target_community, list(newcom.communities), list(self.G.nodes), self.distribute)
        # num_components = nx.number_connected_components(G)
        return eva_score


    # 初始化种群
    def initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            individual = getNewSolutionPriori(self.G, self.num_attack, self.target_community)
            population.append(individual)
        return np.array(population)


    # 变异操作
    def mutate(self, population, F, bounds):
        idx1, idx2, idx3 = np.random.choice(len(population), 3, replace=False)
        mutant = population[idx1] + F * (population[idx2] - population[idx3])

        # 对变异后的结果进行离散化处理
        mutant = np.clip(np.round(mutant), bounds[0], bounds[1]).astype(int)
        return mutant




    # 交叉操作
    def crossover(self, population, mutant, CR):
        crossover_points = np.random.rand(len(population[0])) < CR
        trial = np.where(crossover_points[:, np.newaxis], mutant, population)
        # 确保至少有一个基因是从变异个体来的（引导过程的一部分）
        if not np.any(crossover_points):
            for i in range(len(trial)):
                idx = np.random.randint(0, len(trial[i]))
                trial[i, idx] = mutant[i, idx]
        return trial


    # 选择操作
    def select(self, population, trial):
        fitness_pop = np.array([self.fitness_func(ind) for ind in population])
        fitness_trial = np.array([self.fitness_func(ind) for ind in trial])
        con = fitness_trial >= fitness_pop
        better = np.where(con[:, np.newaxis, np.newaxis], trial, population)
        return better


# 差分进化算法
# def differential_evolution(pop_size, dim, bounds, max_iter, F, CR, fitness_func):
#     population = initialize_population(pop_size, dim, bounds)
#     best_individual = population[np.argmax([fitness_func(ind) for ind in population])]
#     best_fitness = fitness_func(best_individual)
#
#     for _ in range(max_iter):
#         mutants = np.array([mutate(population, F, bounds) for _ in range(pop_size)])
#         trials = crossover(population, mutants, CR)
#         population = select(population, trials, fitness_func)
#
#         current_best = population[np.argmax([fitness_func(ind) for ind in population])]
#         current_best_fitness = fitness_func(current_best)
#
#         if current_best_fitness > best_fitness:
#             best_individual = current_best
#             best_fitness = current_best_fitness
#
#         print(f"Iteration {_}: Best Fitness = {best_fitness}")
#
#     return best_individual, best_fitness


# # 参数设置
# pop_size = 20
# dim = len(target_string)
# bounds = (0, 1)  # 对于离散化，我们将其限制在0和1之间
# max_iter = 100
# F = 0.8  # 缩放因子
# CR = 0.7  # 交叉概率
#
# # 运行差分进化算法
# best_individual, best_fitness = differential_evolution(pop_size, dim, bounds, max_iter, F, CR, fitness_function)
# print(f"Best Individual: {best_individual}, Best Fitness: {best_fitness}")
