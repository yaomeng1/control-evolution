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
fd = 0.05  # fraction of driver nodes
m = 2
control_num = int(nodesnum * fd)


def edge2num(node, scale):
    num = node[0] * scale + node[1]
    return num


def rand_pick(probabilities):
    x = random.uniform(0, 1)
    if x <= probabilities:
        return 1
    else:
        return 0


def first_round(payoff_array, game_matrix, edge_list, state_array):
    for edge in edge_list:
        nodex, nodey = edge
        payoff_array[nodex] += game_matrix[state_array[nodex]][state_array[nodey]]
        payoff_array[nodey] += game_matrix[state_array[nodey]][state_array[nodex]]


def single_round(payoff_array, game_matrix, edge_category, state_array, control_state, pre_state):
    pre_control_state = control_state.copy()
    control2_edge, control1_edge_sort, nocontrol_edge = edge_category
    for edge in nocontrol_edge:
        # no driver node
        nodex, nodey = edge
        payoff_array[nodex] += game_matrix[state_array[nodex]][state_array[nodey]]
        payoff_array[nodey] += game_matrix[state_array[nodey]][state_array[nodex]]
    for edge in control1_edge_sort:
        nodex, nodey = edge
        # nodex is a driver node
        if not pre_state[nodey]:
            # node y is not cooperator in last run
            control_state[(nodex, nodey)] = 1 - pre_control_state[(nodex, nodey)]
        payoff_array[nodex] += game_matrix[control_state[(nodex, nodey)]][state_array[nodey]]
        payoff_array[nodey] += game_matrix[state_array[nodey]][control_state[(nodex, nodey)]]
    for edge in control2_edge:
        # both driver node
        nodex, nodey = edge
        if not pre_control_state[(nodey, nodex)]:
            control_state[(nodex, nodey)] = 1 - pre_control_state[(nodex, nodey)]
        if not pre_control_state[(nodex, nodey)]:
            control_state[(nodey, nodex)] = 1 - pre_control_state[(nodey, nodex)]
        payoff_array[nodex] += game_matrix[control_state[(nodex, nodey)]][control_state[(nodey, nodex)]]
        payoff_array[nodey] += game_matrix[control_state[(nodey, nodex)]][control_state[(nodex, nodey)]]


def replicate_dynamic(payoff_array, adj_dict, state_array, control_state):
    """
    replicator dynamic after single round game

    Whenever a site x is updated, a neighbor y is
    drawn at random among all kx neighbors; whenever Py >Px
    the chosen neighbor takes over site x with probability
    given by (Py-Px)/Dk>, where k> is the largest between
    kx and ky and D = T - S for the PD and D = T -P for the SG.

    """
    previous_state = state_array.copy()
    for node in range(control_num, nodesnum):
        neighbor = random.choice(adj_dict[node])
        prob = 1 / (1 + np.exp(10 * (payoff_array[node] - payoff_array[neighbor])))
        if rand_pick(prob):
            if neighbor < control_num:
                # a control neighbor
                state_array[node] = control_state[(neighbor, node)]
            else:
                state_array[node] = previous_state[neighbor]
    return previous_state


def clear(payoff_dict):
    # 一次性置零
    payoff_dict[:] = 0


def evolution(payoff_array, game_matrix, edge_category, adj_dict,
              state_array, control_state, edge_list):
    """
    whole process of evolution for 10000 times of generation
    """

    freq_list = []
    first_round(payoff_array, game_matrix, edge_list, state_array)
    pre_state = replicate_dynamic(payoff_array, adj_dict, state_array, control_state)
    for _ in range(20000):
        single_round(payoff_array, game_matrix, edge_category, state_array, control_state, pre_state)
        pre_state = replicate_dynamic(payoff_array, adj_dict, state_array, control_state)
        clear(payoff_array)
        coord = np.mean(state_array[control_num:])
        freq_list.append(coord)

    # return np.mean(np.array(freq_list)[-2000:]).reshape(1, -1)
    return np.array(freq_list).reshape(1, -1)



def process(b):
    game_matrix = np.zeros((2, 2))
    game_matrix[0][0] = 0  # P defect--defect
    game_matrix[0][1] = b  # T d-c
    game_matrix[1][0] = 0  # S
    game_matrix[1][1] = 1  # R
    net_list = []
    net_rep = 1
    control_rep = 1
    repeat_time = 1
    for _ in range(net_rep):
        control_list = []
        graph = nx.random_graphs.barabasi_albert_graph(nodesnum, m)
        # graph = nx.random_graphs.random_regular_graph(2 * m, nodesnum)
        # graph = nx.generators.lattice.grid_2d_graph(graph_scale, graph_scale, periodic=True)
        edge_list_ini = [edge for edge in graph.edges()]
        adj_dict_ini = nx.to_dict_of_lists(graph)
        ## lattice
        # edge_list_ini = [(edge2num(edge[0], graph_scale), edge2num(edge[1], graph_scale)) for edge in edge_list_ini]
        # adj_dict = {}
        # for key, item in adj_dict_ini.items():
        #     adj_dict[edge2num(key, graph_scale)] = [edge2num(node, graph_scale) for node in item]
        # adj_dict_ini = adj_dict

        for _ in range(control_rep):
            repeat_list = []
            all_nodes = list(range(nodesnum))
            control_nodes_list = random.sample(all_nodes, int(nodesnum * fd))
            control_nodes_list.sort()
            rest_nodes = set(all_nodes) - set(control_nodes_list)

            control_dict = dict(zip(control_nodes_list, range(control_num)))
            uncontrol_dict = dict(zip(rest_nodes, range(control_num, nodesnum)))
            map_dict = {**uncontrol_dict, **control_dict}
            # renumber control_nodes_list 0~control_num
            edge_list = [(map_dict[edge[0]], map_dict[edge[1]]) for edge in edge_list_ini]
            adj_dict = {map_dict[k]: [map_dict[node] for node in v] for k, v in adj_dict_ini.items()}
            # categorize edge with 2 control nodes, 1control nodes sorted, noconrol
            control2_edge = [edge for edge in edge_list if edge[0] < control_num and edge[1] < control_num]
            nocontrol_edge = [edge for edge in edge_list if edge[0] >= control_num and edge[1] >= control_num]
            rest = set(edge_list) - set(control2_edge) - set(nocontrol_edge)  # 1 control node
            control1_edge_sort = [tuple(sorted(edge)) for edge in rest]  # 1 control node sorted: edge[0]-->control

            for _ in range(repeat_time):
                payoff_array = np.zeros(nodesnum)
                # initialize control_state for control nodes
                control_state = {}
                for i in range(control_num):
                    for neigh in adj_dict[i]:
                        control_state[i, neigh] = 1
                # initialize control_state for normal nodes
                state_array = np.ones(nodesnum, dtype=np.int)
                uncontrolled_select = random.sample(range(control_num, nodesnum),
                                                    int(nodesnum * (1 - fd) / 2))  # half of rest nodes
                for idx_uncon in uncontrolled_select:
                    state_array[idx_uncon] = 0

                repeat_list.append(
                    evolution(payoff_array, game_matrix, (control2_edge, control1_edge_sort, nocontrol_edge), adj_dict,
                              state_array, control_state, edge_list))
                return repeat_list[0]
    #         control_list.append(np.concatenate(repeat_list, axis=1))
    #
    #     net_list.append(np.concatenate(control_list, axis=0).reshape(1, control_rep, repeat_time))
    # return np.concatenate(net_list, axis=0)


if __name__ == "__main__":
    # graph = nx.random_graphs.barabasi_albert_graph(nodesnum, m)
    # graph = nx.generators.lattice.grid_2d_graph(graph_scale, graph_scale, periodic=True)
    #
    # edge_list_ini = [edge for edge in graph.edges()]
    # adj_dict_ini = nx.to_dict_of_lists(graph)
    # # grid transfer
    # edge_list_ini = [(edge2num(edge[0], graph_scale), edge2num(edge[1], graph_scale)) for edge in edge_list_ini]
    # adj_dict = {}
    # for key, item in adj_dict_ini.items():
    #     adj_dict[edge2num(key, graph_scale)] = [edge2num(node, graph_scale) for node in item]
    # adj_dict_ini = adj_dict

    ##----------------debug_test-----------------------------
    # process(b=1.025, edge_list_ini=edge_list_ini, adj_dict_ini=adj_dict_ini)

    ##----------------parallel computation-------------------
    pool = multiprocessing.Pool()
    t1 = time.time()
    # pt = functools.partial(process, edge_list_ini=edge_list_ini, adj_dict_ini=adj_dict_ini)
    # coor_freq = pool.map(pt, b_list)
    coor_freq = pool.map(process, b_list)
    pool.close()
    pool.join()
    t2 = time.time()
    print("Total time:" + (t2 - t1).__str__())
    file = "./fullgene_b_1_2_sf_k4_decentralized_controlrate_5_wsls.pk"
    if not os.path.exists(file):
        os.mknod(file)
    with open('./fullgene_b_1_2_sf_k4_decentralized_controlrate_5_wsls.pk', 'wb') as f:
        pickle.dump([b_list, coor_freq], f)
