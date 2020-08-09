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

b_list = np.arange(1, 1.22, 0.025)
graph_scale = 30
nodesnum = graph_scale ** 2
fd = 0.05  # fraction of driver nodes


def central_controller(adj_dict, controlnum):
    starter = random.choice(range(nodesnum))
    control_nodelist = set([starter])
    control_nodelist_former = set()

    while len(control_nodelist) < controlnum:
        new_add = control_nodelist - control_nodelist_former
        control_nodelist_former = control_nodelist.copy()
        for node in new_add:
            control_nodelist.update(set(adj_dict[node]))

    control_nodelist = list(control_nodelist)[:controlnum]
    return control_nodelist


def edge2num(node, scale):
    num = node[0] * scale + node[1]
    return num


def rand_pick(probabilities):
    x = random.uniform(0, 1)
    if x <= probabilities:
        return 1
    else:
        return 0


def single_round(state_array, payoff_dict, game_matrix, edge_list, control_nodes_list):
    for edge in edge_list:
        nodex, nodey = edge
        payoff_dict[nodex] += game_matrix[state_array[nodex]][state_array[nodey]]
        payoff_dict[nodey] += game_matrix[state_array[nodey]][state_array[nodex]]


def replicate_dynamic(state_dict, payoff_dict, adj_dict, rest_nodes):
    """
    replicator dynamic after single round game

    Whenever a site x is updated, a neighbor y is
    drawn at random among all kx neighbors; whenever Py >Px
    the chosen neighbor takes over site x with probability
    given by (Py-Px)/Dk>, where k> is the largest between
    kx and ky and D = T - S for the PD and D = T -P for the SG.

    """
    previous_state = state_dict.copy()
    for node in rest_nodes:
        neighbor = random.choice(adj_dict[node])
        prob = 1 / (1 + np.exp(10 * (payoff_dict[node] - payoff_dict[neighbor])))
        if rand_pick(prob):
            state_dict[node] = previous_state[neighbor]


def clear(payoff_dict):
    # 一次性置零
    payoff_dict[:] = 0


def evolution(state_array, payoff_dict, game_matrix, edge_list, adj_dict,
              control_bool_array, rest_nodes, control_nodes_list):
    """
    whole process of evolution for 10000 times of generation
    """

    freq_list = []
    for time in range(20000):
        single_round(state_array, payoff_dict, game_matrix, edge_list, control_nodes_list)
        replicate_dynamic(state_array, payoff_dict, adj_dict, rest_nodes)
        clear(payoff_dict)
        coord = np.sum(np.where(control_bool_array, state_array, 0)) / int(nodesnum * (1 - fd))

        freq_list.append(coord)

    return np.mean(np.array(freq_list)[-2000:]).reshape(1, -1)


def process(b, edge_list, adj_dict):
    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0  # P defect--defect
    game_matrix[0][1] = b  # T d-c
    game_matrix[1][0] = 0  # S
    game_matrix[1][1] = 1  # R
    net_list = []
    repeat_time = 50
    net_rep = 10

    for _ in range(net_rep):
        repeat_list = []
        all_nodes = list(range(nodesnum))
        # control_nodes_list = random.sample(all_nodes, int(nodesnum * fd))
        # rest nodes select half
        control_nodes_list = central_controller(adj_dict_ini, control_num)
        rest_nodes = set(all_nodes) - set(control_nodes_list)
        ## centralized

        for _ in range(repeat_time):
            payoff_dict = np.zeros(nodesnum)
            state_array = np.ones(nodesnum, dtype=np.int)
            control_bool_array = np.ones(nodesnum, dtype=np.int)

            # select control nodes --> bool matrix to distinct controlled nodes
            for idx_con in control_nodes_list:
                control_bool_array[idx_con] = 0

            uncontrolled_select = random.sample(rest_nodes, int(nodesnum * (1 - fd) / 2))  # half of rest nodes
            for idx_uncon in uncontrolled_select:
                state_array[idx_uncon] = 0

            repeat_list.append(evolution(state_array, payoff_dict, game_matrix, edge_list, adj_dict,
                                         control_bool_array, rest_nodes, control_nodes_list))
        net_list.append(np.concatenate(repeat_list, axis=1))

    return np.concatenate(net_list, axis=0)


if __name__ == "__main__":
    graph = nx.generators.lattice.grid_2d_graph(graph_scale, graph_scale, periodic=True)
    edge_list_ini = [edge for edge in graph.edges()]
    adj_dict_ini = nx.to_dict_of_lists(graph)

    edge_list = [(edge2num(edge[0], graph_scale), edge2num(edge[1], graph_scale)) for edge in edge_list_ini]
    adj_dict = {}
    for key, item in adj_dict_ini.items():
        adj_dict[edge2num(key, graph_scale)] = [edge2num(node, graph_scale) for node in item]

    ##----------------debug_test-----------------------------
    # process(b=1.1, edge_list=edge_list_ini, adj_dict=adj_dict_ini)

    ##----------------parallel computation-------------------
    pool = multiprocessing.Pool()
    t1 = time.time()
    pt = functools.partial(process, edge_list=edge_list, adj_dict=adj_dict)
    coor_freq = pool.map(pt, b_list)
    # coor_freq = pool.map(process, b_list)
    pool.close()
    pool.join()
    t2 = time.time()
    print("Total time:" + (t2 - t1).__str__())
    file = "./decentralized_controlrate_5_random_same.pk"
    if not os.path.exists(file):
        os.mknod(file)
    with open('./decentralized_controlrate_5_random_same.pk', 'wb') as f:
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
