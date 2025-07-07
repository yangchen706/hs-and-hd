import numpy as np
from unit import *
from cdlib import evaluation,NodeClustering
import math

def get_M1(graph,target_nodelist,current_partitions):
    """
    print the M1 of target nodes， 越大证明隐藏效果越好
    paper: Hiding individuals and communities in a social network
    :param target_nodelist: 目标社区节点list
    :param current_partitions: 当前社区划分list
    :param graph: 原始图 主要用来取所有节点
    :return:
    """
    member_for_community = []
    group_len = 0
    for member in target_nodelist:
        for community in range(len(current_partitions)):
           if member in current_partitions[community]:
              current_community_member=community
              member_for_community.append(current_community_member)
    # print("target_nodelist:",target_nodelist)
    # print(current_partitions)
    print(member_for_community)
    M1=(len(np.unique(member_for_community)) - 1) / (np.max([len(current_partitions) - 1, 1])*(np.max(np.bincount(member_for_community))))

    member_for_communitys = np.unique(member_for_community)
    for idx in member_for_communitys:
        group_len = group_len + len(current_partitions[idx])

    group_len -= len(target_nodelist)
    const_denom = np.max([len(graph.nodes) - len(target_nodelist), 1])
    M2 = group_len / const_denom
    α: float=0.5
    M1 = α * M1
    M2 = α * M2
    M=M1+M2
    return M


def get_M1_overlap(graph,target_nodelist,current_partitions):
    """
    print the M1 of target nodes， 越大证明隐藏效果越好
    paper: Hiding individuals and communities in a social network
    :param target_nodelist: 目标社区节点list
    :param current_partitions: 当前社区划分list
    :param graph: 原始图 主要用来取所有节点
    :return:
    """
    node_in_overlap_community_num = get_num_node_in_communities(current_partitions, list(graph.nodes))
    member_for_community = []
    group_len = set()
    max_intersection_community_target_len = 0
    max_intersection_community_target = []
    intersection_num  = 0
    real_len_of_max_intersection = 0
    for community in range(len(current_partitions)):
        temp_intersection = set(target_nodelist) & set(current_partitions[community])
        if temp_intersection:
            intersection_num += 1
            member_for_community.append(community)
        if len(temp_intersection) > max_intersection_community_target_len:
            max_intersection_community_target_len = len(temp_intersection)
            max_intersection_community_target = temp_intersection
    print(member_for_community)
    # print("target_nodelist:",target_nodelist)
    # print(current_partitions)
    for i in max_intersection_community_target:
        real_len_of_max_intersection += 1/node_in_overlap_community_num[i]
    # print(member_for_community)
    M1=(intersection_num - 1) / (np.max([len(current_partitions) - 1, 1])*real_len_of_max_intersection)

    member_for_communitys = np.unique(member_for_community)
    for idx in member_for_communitys:
        group_len = group_len | set(current_partitions[idx])

    group_len_num = sum([1/node_in_overlap_community_num[i] for i in (group_len - set(target_nodelist))])
    const_denom = np.max([len(graph.nodes) - len(target_nodelist), 1])
    M2 = group_len_num / const_denom
    α: float=0.5
    M1 = α * M1
    M2 = α * M2
    M=M1+M2
    return M


def get_cross_entropy(G,target_commuinty, communities, nodes, distribution):
    '''
    获取标准分布和现有分布的交叉熵，标准分布就是单个社区节点的数量和总节点数的比例组成的分布，
    注：重叠社区因为节点重叠，安装等比值分配到各个社区。否则概率和不为1.
    target_commuinty：目标社区 list类型
    communities：社区划分 list (执行隐藏后)
    nodes：图中所有节点

    返回： 交叉熵的值
    '''
    cross_entroy = 0
    node_in_overlap_community_num = get_num_node_in_communities(communities, nodes)
    # print(node_in_overlap_community_num)

    # nodes_in_overlap = [k for k, v in node_in_overlap_community_num.items() if v > 1]  # 取重叠区域的所有节点

    communities_len = []
    intersection_len = []
    small_p = 0.00001
    p_list = []
    q_list = []
    if distribution == "degree":
        flag = True
    else:
        flag = False

    for i in communities:
        real_len_community = 0  # 社区等比例分配重叠节点权值后的长度
        real_intersection_nodes_in_or_community = 0  # 原始社区和当前社区交集，等比例分配权值后的长度

        for j in i:
            if flag:
                neighbor_i = set(G.neighbors(j))

                real_len_community += len(set(i) & neighbor_i) / len(neighbor_i)
            else:
                real_len_community += 1 / node_in_overlap_community_num[j]
        communities_len.append(real_len_community)

        i_intersection_or_community = set.intersection(set(i), set(target_commuinty))
        for k in i_intersection_or_community:
            if flag:
                neighbor_k = set(G.neighbors(k))
                neighbor_k.add(k)
                real_intersection_nodes_in_or_community += len(set(i_intersection_or_community) & neighbor_k) / len(neighbor_k & set(target_commuinty))

            else:
                real_intersection_nodes_in_or_community += 1 / node_in_overlap_community_num[k]

        intersection_len.append(real_intersection_nodes_in_or_community)

    if len(intersection_len) - intersection_len.count(0) == 0: #防止全部没有交集，分母为0问题
        del_small_p = 0
    else:
        del_small_p = (intersection_len.count(0) * small_p) / (len(intersection_len) - intersection_len.count(0))

    for i in range(len(intersection_len)):
        if intersection_len[i] == 0:
            intersection_len[i] = small_p
        else:
            intersection_len[i] = intersection_len[i] - del_small_p

        p = communities_len[i] / sum(communities_len)
        q = intersection_len[i] / len(target_commuinty)
        cross_entroy += -p * math.log(q, 2)
        p_list.append(p)
        q_list.append(q)
        # print(cross_entroy)
    # print("intersection_len",intersection_len)
    # print("communities_len", communities_len)
    # print("p_list",p_list)
    # print("q_list",q_list)
    return cross_entroy

def get_eva(G,target_commuinty, communities, Gnodes, distribution):
    '''
    结果越小越好
    获取评价函数的值，评价函数有两部分组成，一部分是交叉熵，一部分是连通性评估
    G：networkx 生产的网络对象
    target_commuinty：目标社区 list
    communities：社区结构划分 list
    Gnodes： 图中所有节点
    返回： 评价函数的值 [0，1]
    '''
    # 调整社区数量应对目标社区节点数过小的问题（目标社区节点数小于社区数量）
    target_community_neighbor_nodes = get_target_community_neighbor_nodes(G, target_commuinty)
    neighbor_community = get_neighbor_community(target_community_neighbor_nodes, communities)
    if len(target_commuinty) < len(communities):
        # if len(target_commuinty) < len(neighbor_community):
        #     communities = random.sample(neighbor_community, len(target_commuinty))
        # else:
        communities = neighbor_community

    cross_entropy = get_cross_entropy(G,target_commuinty, communities, Gnodes, distribution)
    # print("cross_entropy:",cross_entropy)

    num_component = len(getConnectedComponentOfCommunity(G, target_commuinty))
    return ((num_component-1) / (len(target_commuinty)-1)) + cross_entropy
    # print("corss:",cross_entropy)
    # print("com:", (num_component-1) / (len(target_commuinty)-1))
    # return cross_entropy




def LONMIScore(G, target_community, communities, commuinties_hiding):
    '''
    局部ONMI，因为对于大图来说对单个社区隐藏对整个网络的结构影响较小，为了更真实的反应对网络的影响，采用局部ONMI
    '''
    target_community_neighbor_nodes = get_target_community_neighbor_nodes(G, target_community)
    target_community_neighbor_community = get_neighbor_community(target_community_neighbor_nodes, communities)
    target_community_neighbor_community_nodes = [k for i in target_community_neighbor_community for k in i]
    communities_after_hiding_of_ONMI = get_communities_after_hiding_of_ONMI(commuinties_hiding, target_community_neighbor_community_nodes)

    target_community_neighbor_community = NodeClustering(target_community_neighbor_community,graph=G,overlap=True)
    communities_after_hiding_of_ONMI = NodeClustering(communities_after_hiding_of_ONMI,graph=G,overlap=True)
    MONMI = ONMIScore(target_community_neighbor_community, communities_after_hiding_of_ONMI)

    return MONMI

