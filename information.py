import networkx as nx
from cdlib import evaluation, algorithms, datasets
import math
import random
from unit import *
import matplotlib.pyplot as plt
import matplotlib as mlp
import numpy as np
from myevaluation import *
import pandas as pd
from multiprocessing import Pool
from dif_evolution import DifferentialEvolution
import psutil
import os
from memory_profiler import profile
import time

@profile
def main(numAttack):
    # 参数设置
    pop_size = 10
    max_iter = 1
    F = 0.8  # 缩放因子
    CR = 0.7  # 交叉概率
    dataname = 'karate'
    detectionFun = "coach"
    target_commuinty_id = 0  # 选择要隐藏的社区
    distribution = "degree" #节点的分配方式 degree 或者 avg
    num = 0  # 计数器记录循环的次数

    G = getG(dataname)

    GNodes = list(G.nodes)  # 获取图的所有节点
    GEdges = list(G.edges)
    bounds = (0, len(GNodes)-1)  # 对于离散化，我们将其限制在0和len(GNodes)之间

    com = getCommunity(G, detectionFun)  # 检测出的社区列表
    or_communities = sorted([sorted(i) for i in com.communities])

    target_community = or_communities[target_commuinty_id]

    d_e = DifferentialEvolution(G, detectionFun, target_community, numAttack, distribution) #初始化差分进化的类

    init_pop = d_e.initialize_population(pop_size)  # 初始化种群

    best_individual = init_pop[np.argmin([d_e.fitness_func(ind) for ind in init_pop])]
    best_fitness = d_e.fitness_func(best_individual)

    start = time.time()  # 差分进化开始时间

    for _ in range(max_iter):
        mutants = np.array([d_e.mutate(init_pop, F, bounds) for _ in range(pop_size)])
        #变异最优值引导
        if best_individual is not None:
            guidance_vector = np.clip(best_individual[np.newaxis, :] - init_pop, bounds[0], bounds[1]).astype(int)
            mutants = np.clip(mutants + guidance_vector, bounds[0], bounds[1]).astype(int)

        trials = d_e.crossover(init_pop, mutants, CR)
        population = d_e.select(init_pop, trials)

        current_best = population[np.argmin([d_e.fitness_func(ind) for ind in population])]
        current_best_fitness = d_e.fitness_func(current_best)

        if current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness

        best_individual_list = [list(i) for i in best_individual]
        print(f"Iteration {_}: Best Fitness = {best_fitness}, best_individual={best_individual_list}")

    end = time.time()#差分进化结束时间
    print(f"耗时: {end - start:.4f} 秒")

    hiding_G = newG(best_individual,G)
    hiding_communities = getCommunity(hiding_G, detectionFun)

    LONMI = LONMIScore(G, target_community, or_communities, hiding_communities.communities)

    print('LONMI:', LONMI)



if __name__ == "__main__":
    # pool = Pool()
    #
    # for i in range(1,101):
    #     pool.apply_async(func=main, args=(9,))  # 维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    #
    # pool.close()
    # pool.join()

    main(8)






