import networkx as nx
from networkx.algorithms import community
import random
from cdlib import evaluation, algorithms
import igraph as ig
import numpy as np


def getOverlappingNode(com):
    for i in range(len(com)):
        for j in com:
            pass

def getG(dataname, type='networkx'):
    '''
    获取networkx图结构
    https://public.websites.umich.edu/~mejn/netdata/
        karate:卡特俱乐部(N:34 E:78)； dolphins：海豚（N:62 E:159）; lesmis:悲惨世界人物(N:77 E:254),
        netscience:网络理论和实验科学家的合著网络(N:1589 E:2742);cond-mat-2005:2005版预印本合作网络（大）N:40421 E:175692
        power:美国西部电力网络(N:4941 E:6594); football:(N:115 E:613)

    http://snap.stanford.edu/data/email-Enron.html
    Email-Enron:欧洲邮件网络（大） http://snap.stanford.edu/data/email-Enron.html(N:36692 E:183831)
    deezer:Social network of Deezer users from Europe. http://snap.stanford.edu/data/feather-deezer-social.html(N:28281 E:92752)
    lastfm_asia:A social network of LastFM users   http://snap.stanford.edu/data/feather-lastfm-social.html
    '''
    data_gml = ['karate', 'dolphins', 'football', 'power','adjnoun', 'lesmis', 'netscience', 'cond-mat-2005']
    data_edgelist = ['dblp', 'deezer', 'geom', 'Email-Enron', 'lastfm_asia']
    if dataname in data_gml:
        g = nx.read_gml('./datasets/'+dataname+'.gml', label="id")
    if dataname in data_edgelist:
        g = nx.read_edgelist('./datasets/'+dataname+'.edgelist', nodetype=int)

    g = g.to_undirected()  # 简化图 改实验仅仅对无向图操作
    if not nx.is_connected(g): #清除孤立的节点
        max_comp = max(nx.connected_components(g), key=len)
        g = nx.subgraph(g, max_comp)
    # 创建一个映射字典
    mapping = {old_id: new_id for new_id, old_id in enumerate(g.nodes())}
    # 使用 relabel_nodes 函数重新标记节点
    H = nx.relabel_nodes(g, mapping)
    if type == "igraph":
        H = ig.Graph.from_networkx(g)
    return H

def getCommunity(G,detectionFun='k_clique_communities'):
    '''
    获取总的图和社区
    filename:文件名
    detectionFun：社区检测方法(使用的是networkX中的方法)
    返回：
    G：newworkx图结构
    comList:社区list
    '''
    if detectionFun == 'CPM':
        com = algorithms.kclique(G, k=4) #K集团方法，用于重叠社区划分，需要输入最少社区划分数量
    if detectionFun == "lais2":
        com = algorithms.lais2(G)
    if detectionFun == "umstmo":
        com = algorithms.umstmo(G)
    if detectionFun == "coach":
        com = algorithms.coach(G)
    if detectionFun == "lfm":
        com = algorithms.lfm(G,0.8)
    if detectionFun == "slpa":
        #40 second, 不稳定
        com = algorithms.slpa(G)
    if detectionFun == "percomvc":
        # 80 second 选择
        com = algorithms.percomvc(G)
    if detectionFun == "demon":
        com = algorithms.demon(G, min_com_size=3, epsilon=0.25)
    if detectionFun == "ego_networks":
        #10second 稳定 无论文
        com = algorithms.ego_networks(G)
    if detectionFun == "ipca":
        com = algorithms.ipca(G)
    #上面都是重叠的
    if detectionFun == "gre":
        com = algorithms.greedy_modularity(G)
    if detectionFun == "informap":
        com = algorithms.infomap(G)
    if detectionFun == "girvan_newman":
        #namely eb in our paper
        com = algorithms.girvan_newman(G,level=3)
    if detectionFun == "lp":
    	com = algorithms.label_propagation(G)
    if detectionFun == "ga":
    	com = algorithms.ga(G)
    if detectionFun == "walktrap":
        com = algorithms.walktrap(G)


    return com

def getCommunityOfList(G,detectionFun='k_clique_communities'):
    '''
    因为很多社区算法返回的是生成器
    所以将社区数据结构转换为list
    '''
    comList = []
    orComList = []
    com = getCommunity(G,detectionFun)
    for g in com :
        orComList.append(list(g))
        if len(g) > 2:
            comList.append(list(g))

    return comList,orComList

def iniCom(G,c):
    '''
    初始化社区，返回社区内部节点的度，点在社区内的边，点在社区外部的边
    G:网络图
    c:需要初始化的社区
    '''
    ini = {}  # 用于保存要社区各个节点的度，社区内节点内部边的数量，社区内节点外部边的数量
    for n in c:
        NE = 0  # 初始化内部边数量
        OE = 0  # 初始化外部边数量
        deg = G.degree(n)
        neighbors = list(nx.all_neighbors(G, n))

        for nei in neighbors:
            if nei in c:
                NE = NE + 1
            else:
                OE = OE + 1

        ini[n] = [deg, NE, OE]

    return ini

def getConnectedComponentOfCommunity(G,c):
    '''
    获取社区c的连通分量的节点list
    '''
    CCList = []
    booleanOfC = {}
    #初始化一个和c等长的一个布尔字典用来记录节点的访问情况
    for i in c:
        booleanOfC[i] = False

    H = G.subgraph(c).copy()
    for i in c:
        if not booleanOfC[i]:
            nList = list(nx.dfs_preorder_nodes(H, source=i))  # 深度节点遍历
            CCList.append(nList)
            for i in nList:
                booleanOfC[i] = True

    return CCList


def getDeg(G,c):
    '''
    获取社区内的度序列
    返回以label为键的字典
    '''
    DegDic = {}
    for n in c:
        DegDic[n] = G.degree(n)

    return DegDic

def getBetweennessCentrality(G,c):
    SG = G.subgraph(c.copy())
    BC = nx.algorithms.centrality.betweenness_centrality(SG,normalized=False)
    return BC

def getCommonNeighbor(G,c,chooseNode):
    #获取社区内目标节点和邻居节点间 公共邻居的个数
    CN = {}
    H = G.subgraph(c)
    # neighborChooseNode = nx.all_neighbors(H,chooseNode)
    for n in c:
        CN[n] = len(list(nx.common_neighbors(H,chooseNode,n)))

    return CN

def ONMIScore(orcom,com):
    """
    获取两个社区的ONMI，注意这里的orcom和com是cdlib的聚类
    """
    return evaluation.overlapping_normalized_mutual_information_LFK(orcom,com).score #计算两个社区的ONMI

def getOverlappingNode(com):
    '''
    获取重叠社区检测算法中的所有重叠的节点
    com:划分社区后节点list
    '''
    lenOfCom = len(com)
    z = set()
    for i in range(lenOfCom-1):
        for j in range(i+1,lenOfCom):
            if set(com[i]) & set(com[j]):
                z = z.union(set(com[i]) & set(com[j]))

    z = list(z)
    z.sort()
    return z

def get_num_node_in_communities(communities,nodes):
    '''
    以字典的形式获取所有节点在各个社区中出现的次数，可以用来排定那些节点是在重叠区域的
    input:communities 社区划分 list形式
    nodes：图中的所有节点
    返回：
        字典：所有节点在各个社区中出现的次数(不包含没有在社区结构中出现的节点)
    '''
    num_node_in_communities = {}
    k = 0
    for i in nodes:
        for j in communities:
            if i in j:
                k += 1

        if k > 0:
            num_node_in_communities[i] = k
        k = 0
    return num_node_in_communities


def newG(edgesOfChange,G):
    """
    根据边list对图进行修改并返回
    """
    H = G.copy()
    # print("edgesOfChange",list(edgesOfChange))
    for i in edgesOfChange:
        if str(i[0]) != str(i[1]):
            i = tuple(i)
            if i in list(H.edges):
                H.remove_edge(*i)
            else:
                H.add_edge(*i)
    return H

def getIntarEdgeCommunities(G,target_community):
    """
    返回目标社区内可以删除的边
    """

    subGraph = G.subgraph(target_community)
    IntarEdges = subGraph.edges

    return IntarEdges

def getInterEdgeCommunities(G,GNodes,target_community,num):
    """
    返回社区间 可以增加的边
    """
    addEdges = []
    target_community_neighbor_nodes = get_target_community_neighbor_nodes(G, target_community)
    different_nodes = list(set(GNodes) - set(target_community_neighbor_nodes))
    edges_list = list(G.edges)
    while True:
        n1 = random.sample(target_community,1)[0]
        n2 = random.sample(different_nodes,1)[0]
        now_edge = tuple(sorted([n1,n2]))
        if now_edge not in addEdges and (n1, n2) not in edges_list and (n2, n1) not in edges_list :
            addEdges.append((n1,n2))
        if len(addEdges) >= num:
            break
    return addEdges

def getNewSolutionPriori(G, numAttack, target_community):

    # del_edges_num = random.randint(1,numAttack)
    IntarEdges = getIntarEdgeCommunities(G,target_community)
    addEdges = getInterEdgeCommunities(G, list(G.nodes),target_community,len(IntarEdges))
    newSolution = sorted(random.sample(list(IntarEdges) + addEdges, numAttack))

    return newSolution
    

    



def get_target_community_neighbor_nodes(G, target_community):
    '''
    获取目标社区的邻居节点和目标社区的节点组成的list
    G：原始社区 networkx
    target_commuinty: 目标社区
    '''
    target_community_neighbor_nodes = []
    for i in target_community:
        i_neighbor = nx.all_neighbors(G, i)
        target_community_neighbor_nodes += list(set(i_neighbor) | set(target_community))
    return list(set(target_community_neighbor_nodes))


def get_neighbor_community(target_community_neighbor_nodes, communities):
    '''
    与目标社区中的成员有直接连接的社区都是目标社区的邻居
    target_community_neighbor_nodes:目标社区邻居节点 list
    ommunities:社区结构 list
    '''
    target_community_neighbor_community = []
    for k in communities:
        if set(k) & set(target_community_neighbor_nodes):
            target_community_neighbor_community.append(k)

    return target_community_neighbor_community


def get_communities_after_hiding_of_ONMI(commuinties_hiding, target_community_neighbor_community_nodes):
    '''
    返回执行隐藏算法后和原目标社区相关的邻居节点和目标社区节点组成的社区结构
    commuinties_hiding：执行隐藏算法后的社区划分
    target_community_neighbor_community_nodes：目标社区节点和目标社区的邻居所在社区所有的节点
    '''
    communities_neighbor_after_hiding = []
    for i in commuinties_hiding:
        if set(i) & set(target_community_neighbor_community_nodes):
            communities_neighbor_after_hiding.append(list(set(i) & set(target_community_neighbor_community_nodes)))
    return communities_neighbor_after_hiding



