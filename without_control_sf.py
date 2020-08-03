import networkx as nx
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import multiprocessing
from multiprocessing import Process, Manager
import functools
import time

b_list = np.arange(1, 2, 0.1)
graph_scale = 30
nodesnum = graph_scale ** 2
m = 3



def rand_pick(probabilities):
    x = random.uniform(0, 1)
    if x <= probabilities:
        return 1
    else:
        return 0


def single_round(state_dict, payoff_dict, game_matrix, edge_list):
    """

    :param state_dict:
    :param payoff_dict:
    :param game_matrix:
    :param edge_list:
    :param control_nodes: bool matrix
    :param control_state: a driver num* 2 *2 ndarray
    :param control_dict: key:(position tuple in lattice), value: range(driver num)
    :return:
    """

    for edge in edge_list:
        nodex, nodey = edge
        payoff_dict[nodex] += game_matrix[state_dict[nodex]][state_dict[nodey]]
        payoff_dict[nodey] += game_matrix[state_dict[nodey]][state_dict[nodex]]


def replicate_dynamic(state_dict, payoff_dict, adj_dict):
    """
    replicator dynamic after single round game

    Whenever a site x is updated, a neighbor y is
    drawn at random among all kx neighbors; whenever Py >Px
    the chosen neighbor takes over site x with probability
    given by (Py-Px)/Dk>, where k> is the largest between
    kx and ky and D = T - S for the PD and D = T -P for the SG.

    """
    prevoius_state = state_dict.copy()
    for node in adj_dict.keys():
        neighbor = random.choice(adj_dict[node])
        prob = 1 / (1 + np.exp(10 * (payoff_dict[node] - payoff_dict[neighbor])))
        if rand_pick(prob):
            state_dict[node] = prevoius_state[neighbor]


def clear(payoff_dict):
    # 一次性置零
    payoff_dict[:] = 0


def evolution(state_dict, payoff_dict, game_matrix, edge_list, adj_dict):
    """
    whole process of evolution for 10000 times of generation
    """

    freq_list = []
    for time in range(20000):
        single_round(state_dict, payoff_dict, game_matrix, edge_list)
        replicate_dynamic(state_dict, payoff_dict, adj_dict)
        clear(payoff_dict)
        coord = np.sum(state_dict) / nodesnum
        freq_list.append(coord)

    return np.mean(np.array(freq_list)[-2000:]).reshape(1, -1)


def process(b):
    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0  # P defect--defect
    game_matrix[0][1] = b  # T d-c
    game_matrix[1][0] = 0  # S
    game_matrix[1][1] = 1  # R
    net_list = []
    net_rep = 10
    repeat_time = 50
    for _ in range(net_rep):
        repeat_list = []
        graph = nx.random_graphs.barabasi_albert_graph(nodesnum, m)
        # graph = nx.random_graphs.random_regular_graph(2 * m, nodesnum)
        # graph = nx.generators.lattice.grid_2d_graph(graph_scale, graph_scale, periodic=True)
        edge_list = [edge for edge in graph.edges()]
        adj_dict = nx.to_dict_of_lists(graph)
        ## lattice
        # edge_list_ini = [(edge2num(edge[0], graph_scale), edge2num(edge[1], graph_scale)) for edge in edge_list_ini]
        # adj_dict = {}
        # for key, item in adj_dict_ini.items():
        #     adj_dict[edge2num(key, graph_scale)] = [edge2num(node, graph_scale) for node in item]
        # adj_dict_ini = adj_dict
        for _ in range(repeat_time):
            payoff_dict = np.zeros(nodesnum)
            state_dict = np.zeros(nodesnum, dtype=np.int)

            # select control nodes --> bool matrix to distinct controlled nodes
            all_nodes = list(range(nodesnum))
            pos_nodes_list = random.sample(all_nodes, int(nodesnum / 2))

            for idx_uncon in pos_nodes_list:
                state_dict[idx_uncon] = 1

            repeat_list.append(evolution(state_dict, payoff_dict, game_matrix, edge_list, adj_dict))
        net_list.append(np.concatenate(repeat_list, axis=1))
    return np.concatenate(net_list, axis=0)


if __name__ == "__main__":
    # graph = nx.generators.lattice.grid_2d_graph(graph_scale, graph_scale, periodic=True)
    # edge_list_ini = [edge for edge in graph.edges()]
    # adj_dict_ini = nx.to_dict_of_lists(graph)

    ##----------------debug_test-----------------------------
    # process(b=1.1, edge_list=edge_list_ini, adj_dict=adj_dict_ini)

    ##----------------parallel computation-------------------
    pool = multiprocessing.Pool()
    t1 = time.time()
    # pt = functools.partial(process, edge_list=edge_list_ini, adj_dict=adj_dict_ini)
    # coor_freq = pool.map(pt, b_list)
    coor_freq = pool.map(process, b_list)
    pool.close()
    pool.join()
    t2 = time.time()
    print("Total time:" + (t2 - t1).__str__())
    file = "./b_1_2_sf_k6_coor_freq_without_control.pk"
    if not os.path.exists(file):
        os.mknod(file)
    with open('./b_1_2_sf_k6_coor_freq_without_control.pk', 'wb') as f:
        pickle.dump([b_list, coor_freq], f)

    ## -------------------draw the graph----------------------
    # pos = nx.spring_layout(graph, iterations=800)

    # node_dict = {(i, j): 0 for i in range(graph_scale) for j in range(graph_scale)}
    # for i in range(8, 13):
    #     for j in range(6, 14):
    #         # for i in range(10,10+9):
    #         #     for j in range(10,10+10):
    #         node_dict[i, j] = 1
    # for com in range(2):
    #     list_nodes = [nodes for nodes in node_dict.keys() if node_dict[nodes] == com]
    #     nx.draw_networkx_nodes(graph, pos, list_nodes, node_size=20, node_color=["k","r"][com])

    # nx.draw(graph, pos, with_labels=False, node_size=20)
    # plt.show()
