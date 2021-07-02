import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import argparse
import time
import pickle
from os import path
import networkx as nx
from sklearn.model_selection import train_test_split, KFold
import sklearn.svm
import pandas as pd

from GROMOV_personal import Generalisation_OT
from GROMOV_personal import entropic_gromov_wasserstein
from GROMOV_personal import compute_distance
from GROMOV_personal import compute_L
from GROMOV_personal import compute_distance_sparse
import GROMOV_personal as Gromov


import S_GWL_Toolkit as GwGt
import ot
from Sliced_GW import sgw_cpu


def get_C(Xs, Xt, function=False, dimension=1, enough_space=True, square_root=True, GPU=False):
    if function:
        if enough_space:
            C1, C2 = get_C(Xs, Xt, square_root=square_root)

            def C1_(array_ij, array_kl=None):
                a = C1[array_ij[0], array_kl]
                return a.reshape(len(array_ij[0]), -1, 1)

            def C2_(array_ij, array_kl=None):
                a = C2[array_ij[0], array_kl]
                return a.reshape(len(array_ij[0]), 1, -1)

            return C1_, C2_

        else:
            raise Exception("not totally implemented")
            if square_root:
                def C1_(array_ij):
                    a = np.sum((Xs[array_ij[0], np.newaxis, :] - Xs[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
                    return a[:, :, np.newaxis]

                def C2_(array_ij):
                    a = np.sum((Xt[array_ij[0], np.newaxis, :] - Xt[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
                    return a[:, np.newaxis, :]
            else:
                def C1_(array_ij):
                    a = np.sum((Xs[array_ij[0], np.newaxis, :] - Xs[np.newaxis, :, :]) ** 2, axis=2)
                    return a[:, :, np.newaxis]

                def C2_(array_ij):
                    a = np.sum((Xt[array_ij[0], np.newaxis, :] - Xt[np.newaxis, :, :]) ** 2, axis=2)
                    return a[:, np.newaxis, :]

            return C1_, C2_

    else:
        if square_root:
            C1 = np.sum((Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
            C2 = np.sum((Xt[:, np.newaxis, :] - Xt[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
        else:
            C1 = np.sum((Xs[:, np.newaxis, :] - Xs[np.newaxis, :, :]) ** 2, axis=2)
            C2 = np.sum((Xt[:, np.newaxis, :] - Xt[np.newaxis, :, :]) ** 2, axis=2)
        return C1, C2


def create_dataset(args, name_dataset="gaussian", n_samples_s=20, n_samples_t=20, same_s_and_t=False,
                   dimension_s=2, dimension_t=1):
    np.random.seed(args.rdm_seed)
    Cs, Xs, Ys, Ct, Xt, Yt = None, None, None, None, None, None
    if name_dataset == "gaussian":
        mu_s = np.array([0, 0])
        cov_s = np.array([[1, 0], [0, 1]])

        mu_t = mu_s
        cov_t = cov_s

        Xs = np.random.multivariate_normal(mu_s, cov_s, n_samples_s)
        Xt = np.random.multivariate_normal(mu_t, cov_t, n_samples_t)

        if "sampled_gromov" in args.name_algo:
            Cs, Ct = get_C(Xs, Xt, function=True,
                           dimension=args.dimension,
                           enough_space=args.enough_space)
        else:
            Cs, Ct = get_C(Xs, Xt)

    elif name_dataset == "uniform":
        Xs = np.random.uniform(0, 1, (n_samples_s, dimension_s))
        Xt = np.random.uniform(0, 1, (n_samples_t, dimension_t))

        if "sampled_gromov" in args.name_algo:
            Cs, Ct = get_C(Xs, Xt, function=True,
                           dimension=args.dimension,
                           enough_space=args.enough_space)
        else:
            Cs, Ct = get_C(Xs, Xt)

    elif name_dataset == "gaussian_graph":

        # generate synthetic graph
        Gs = nx.gaussian_random_partition_graph(n=n_samples_s, s=args.clique_size_s,
                                                v=args.variance_s,
                                                p_in=args.p_in_t, p_out=args.ps_out_s,
                                                directed=False,
                                                seed=np.random.randint(10000))

        Ys = np.zeros((n_samples_s,))
        for i in range(len(Gs.nodes)):
            Ys[i] = Gs.nodes[i]['block']

        Gt = nx.gaussian_random_partition_graph(n=n_samples_t, s=args.clique_size_t,
                                                v=args.variance_t,
                                                p_in=args.p_in_t, p_out=args.ps_out_t,
                                                directed=False,
                                                seed=np.random.randint(10000))

        Yt = np.zeros((n_samples_s,))
        for i in range(len(Gs.nodes)):
            Yt[i] = Gs.nodes[i]['block']

        if "sampled_gromov" in args.name_algo:

            if args.enough_space:
                Cs_matrix, Ct_matrix = nx.adjacency_matrix(Gs).toarray(), nx.adjacency_matrix(Gt).toarray()
                if args.value_graph:
                    # TODO not very efficient if the graph is sparse
                    Cs_matrix = Cs_matrix * np.random.uniform(0.25, 1.75, Cs_matrix.shape)
                    Ct_matrix = Ct_matrix * np.random.uniform(0.25, 1.75, Ct_matrix.shape)

                Cs_matrix += Cs_matrix.T
                Ct_matrix += Ct_matrix.T

                # print(Ct_matrix, Ct_matrix)
                def Cs(array_ij, array_kl=None):
                    a = Cs_matrix[array_ij[0], array_kl]
                    return a.reshape(len(array_ij[0]), -1, 1)

                def Ct(array_ij, array_kl=None):
                    a = Ct_matrix[array_ij[0], array_kl]
                    return a.reshape(len(array_ij[0]), 1, -1)

            else:
                raise Exception("Not implemented yet, use enough_space = True")

                def Cs(array_ij):
                    a = nx.adjacency_matrix(Gs)
                    return a[:, :, np.newaxis]

                def Ct(array_ij):
                    a = np.sum((Xt[array_ij[0], np.newaxis, :] - Xt[np.newaxis, :, :]) ** 2, axis=2) ** 0.5
                    return a[:, np.newaxis, :]

        else:
            Cs, Ct = nx.adjacency_matrix(Gs).toarray(), nx.adjacency_matrix(Gt).toarray()
            Cs = Cs * np.random.uniform(0.25, 1.75, Cs.shape)
            Ct = Ct * np.random.uniform(0.25, 1.75, Ct.shape)
            Cs += Cs.T
            Ct += Ct.T
    elif name_dataset == "gaussian_point_graph":

        args.ps_out_s = 1 / args.ps_out_s
        args.p_in_s = 1 / args.p_in_s
        args.ps_out_t = 1 / args.ps_out_t
        args.p_in_t = 1 / args.p_in_t

        def create_X(dimension, clique_size, ps_out, p_in, variance, n_samples):
            X = None
            number_points_total = 0
            while True:

                center_gaussian = np.random.multivariate_normal(np.zeros(dimension),
                                                                np.eye(dimension) * ps_out)
                number_points = int(np.random.normal(clique_size, variance))
                number_points = np.clip(number_points, 0, n_samples - number_points_total)

                new_points = np.random.multivariate_normal(center_gaussian,
                                                           np.eye(dimension) * p_in,
                                                           number_points)
                if X is None:
                    X = new_points
                else:
                    X = np.concatenate(([X, new_points]), axis=0)
                number_points_total += number_points
                if n_samples <= number_points_total:
                    break
            return X

        Xs = create_X(dimension=dimension_s,
                      clique_size=args.clique_size_s,
                      ps_out=args.ps_out_s,
                      p_in=args.p_in_s,
                      variance=args.variance_s,
                      n_samples=n_samples_s)
        Xt = create_X(dimension=dimension_t,
                      clique_size=args.clique_size_t,
                      ps_out=args.ps_out_t,
                      p_in=args.p_in_t,
                      variance=args.variance_t,
                      n_samples=n_samples_t)

        if "sampled_gromov" in args.name_algo:
            Cs, Ct = get_C(Xs, Xt, function=True,
                           dimension=args.dimension,
                           enough_space=args.enough_space,
                           square_root=False)  # to allow a fair comparaison with sliced
        else:
            Cs, Ct = get_C(Xs, Xt, square_root=False)  # to allow a fair comparaison with sliced
    elif name_dataset == "same_graph":
        # generate synthetic graph
        Gs = nx.gaussian_random_partition_graph(n=n_samples_s, s=args.clique_size_s,
                                                v=args.variance_s,
                                                p_in=args.p_in_t, p_out=args.ps_out_s,
                                                directed=True,
                                                seed=np.random.randint(10000))

        # TODO this is horrible... but to_directed() and directed=True doesn't work...
        Cs_temp = nx.to_numpy_array(Gs)
        Cs_temp = ((Cs_temp + Cs_temp.T) != 0) * 1  # Symmetric
        Ct_temp = ((Cs_temp + (
                np.random.rand(Cs_temp.shape[0], Cs_temp.shape[1]) < args.noise_graph)) >= 1) * 1  # add noise
        Gt = nx.from_numpy_array(Ct_temp)
        Gs = nx.from_numpy_array(Cs_temp)
        Gt.remove_edges_from(nx.selfloop_edges(Gt))  # delete self loop edge due to noise

        if "sampled_gromov" in args.name_algo:
            if args.enough_space:
                shortest_path = nx.shortest_path_length(Gs, source=None, target=None)
                Cs_matrix = np.empty((len(Gs.nodes), len(Gs.nodes)))
                for dict_i in shortest_path:
                    for key in dict_i[1]:
                        Cs_matrix[dict_i[0], key] = dict_i[1][key]

                shortest_path = nx.shortest_path_length(Gt, source=None, target=None)
                Ct_matrix = np.empty((len(Gt.nodes), len(Gt.nodes)))
                for dict_i in shortest_path:
                    for key in dict_i[1]:
                        Ct_matrix[dict_i[0], key] = dict_i[1][key]

                def Cs(array_ij, array_kl=None):
                    a = Cs_matrix[array_ij[0], array_kl]
                    return a.reshape(len(array_ij[0]), -1, 1)

                def Ct(array_ij, array_kl=None):
                    a = Ct_matrix[array_ij[0], array_kl]
                    return a.reshape(len(array_ij[0]), 1, -1)
            else:

                def Cs(array_ij, array_kl=None):
                    global Cs_save
                    # if array_kl is None:
                    for pos_i in array_ij[0]:
                        if np.isnan(Cs_save[pos_i, 0]):
                            shortest_path = nx.shortest_path_length(Gs, source=pos_i, target=None)
                            for key in shortest_path:
                                Cs_save[pos_i, key] = shortest_path[key]
                    a = Cs_save[array_ij[0], array_kl]
                    return a.reshape(len(array_ij[0]), -1, 1)

                def Ct(array_ij, array_kl=None):
                    global Ct_save
                    for pos_i in array_ij[0]:
                        if np.isnan(Ct_save[pos_i, 0]):
                            shortest_path = nx.shortest_path_length(Gs, source=pos_i, target=None)
                            for key in shortest_path:
                                Ct_save[pos_i, key] = shortest_path[key]
                    a = Ct_save[array_ij[0], array_kl]
                    return a.reshape(len(array_ij[0]), 1, -1)

        else:
            shortest_path = nx.shortest_path_length(Gs, source=None, target=None)
            Cs = np.empty((len(Gs.nodes), len(Gs.nodes)))
            for dict_i in shortest_path:
                for key in dict_i[1]:
                    Cs[dict_i[0], key] = dict_i[1][key]

            shortest_path = nx.shortest_path_length(Gt, source=None, target=None)
            Ct = np.empty((len(Gt.nodes), len(Gt.nodes)))
            for dict_i in shortest_path:
                for key in dict_i[1]:
                    Ct[dict_i[0], key] = dict_i[1][key]
    elif name_dataset == "reddit_binary":  # Uniform is the best... useless dataset.
        if os.path.exists("./DATA/REDDIT-BINARY/C_label.pickle"):
            with open("./DATA/REDDIT-BINARY/C_label.pickle", "rb") as f:
                C_label = pickle.load(f)
            return C_label["C"], C_label["label"]
        else:
            with open("./DATA/REDDIT-BINARY/REDDIT-BINARY.edges", "r") as edges, \
                    open("./DATA/REDDIT-BINARY/REDDIT-BINARY.graph_idx", "r") as graph_idx, \
                    open("./DATA/REDDIT-BINARY/REDDIT-BINARY.graph_labels", "r") as graph_labels:
                graph_labels_list = graph_labels.readlines()
                n = len(graph_labels_list)
                C = [0] * n
                print("Load data")
                graph_idx_i = int(graph_idx.readline())
                for i in range(1, n + 1):
                    if i % 100 == 0:
                        print(".", end="")
                    G = nx.DiGraph()
                    while i == graph_idx_i:
                        edge = edges.readline()
                        edge = edge.split(",")
                        G.add_edge(int(edge[0]), int(edge[1]))
                        try:
                            graph_idx_i = int(graph_idx.readline())
                        except:
                            assert i == n
                            break
                    shortest_path = nx.shortest_path_length(G, source=None, target=None)
                    n_node = len(G.nodes)
                    C_ = np.ones((n_node, n_node)) * 25  # maximum value observed in the dataset
                    mapping = dict(zip(G.nodes, range(len(G.nodes))))
                    for dict_i in shortest_path:
                        for key in dict_i[1]:
                            C_[mapping[dict_i[0]], mapping[key]] = dict_i[1][key]
                    C[i - 1] = C_

            with open("./DATA/REDDIT-BINARY/C_label.pickle", "wb") as f:
                pickle.dump({"C": C, "label": graph_labels_list}, f)
            print("")
            print("Data loaded and saved")
            return C, graph_labels_list
    elif name_dataset == "DB":
        if os.path.exists("./DATA/FIRSTMM_DB/C_label.pickle"):
            with open("./DATA/FIRSTMM_DB/C_label.pickle", "rb") as f:
                C_label = pickle.load(f)

            return C_label["C"], C_label["label"]
        else:
            with open("./DATA/FIRSTMM_DB/FIRSTMM_DB_A.txt", "r") as edges, \
                    open("./DATA/FIRSTMM_DB/FIRSTMM_DB_graph_indicator.txt", "r") as graph_idx, \
                    open("./DATA/FIRSTMM_DB/FIRSTMM_DB_graph_labels.txt", "r") as graph_labels:
                graph_labels_list = graph_labels.readlines()
                n = len(graph_labels_list)
                C = [0] * n
                print("Load data")
                graph_idx_i = int(graph_idx.readline())
                for i in range(1, n + 1):
                    if i % 100 == 0:
                        print(".", end="")
                    G = nx.DiGraph()
                    while i == graph_idx_i:
                        edge = edges.readline()
                        edge = edge.split(",")
                        G.add_edge(int(edge[0]), int(edge[1]))
                        try:
                            graph_idx_i = int(graph_idx.readline())
                        except:
                            assert i == n
                            break
                    shortest_path = nx.shortest_path_length(G, source=None, target=None)
                    n_node = len(G.nodes)
                    C_ = np.ones((n_node, n_node)) * (-1)  # to later set -1 to the max value.
                    mapping = dict(zip(G.nodes, range(len(G.nodes))))
                    for dict_i in shortest_path:
                        for key in dict_i[1]:
                            C_[mapping[dict_i[0]], mapping[key]] = dict_i[1][key]
                    C_[C_ == -1] = np.max(C_) + 1
                    C[i - 1] = C_
            with open("./DATA/FIRSTMM_DB/C_label.pickle", "wb") as f:
                pickle.dump({"C": C, "label": graph_labels_list}, f)
            print("")
            print("Data loaded and saved")
            return C, graph_labels_list

    else:
        raise Exception("Wrong name of dataset")
    if same_s_and_t:
        Ct, Xt, Yt = Cs, Xs, Ys

    return Cs, Xs, Ys, Ct, Xt, Yt


def define_loss_function(loss_func_name, GPU=False):
    if loss_func_name == "square_loss":
        def loss_fun(C1, C2):
            return (C1 - C2) ** 2
    elif loss_func_name == "2_loss":
        def loss_fun(C1, C2):
            return (C1 - C2) ** 2
    elif loss_func_name == "3_loss":
        def loss_fun(C1, C2):
            return np.abs(C1 - C2) ** 3
    elif loss_func_name == "15_loss":
        def loss_fun(C1, C2):
            return np.abs(C1 - C2) ** 1.5
    elif loss_func_name == "25_loss":
        def loss_fun(C1, C2):
            return np.abs(C1 - C2) ** 2.5
    elif loss_func_name == "1_loss":
        def loss_fun(C1, C2):
            return np.abs(C1 - C2)
    elif loss_func_name == "2exp_loss":
        def loss_fun(C1, C2):
            return 1 - np.exp(-(C1 - C2) ** 2)
    elif loss_func_name == "2exp_loss_sigma":
        def loss_fun(C1, C2):
            return 1 - np.exp(((C1 - C2) / 10) ** 2)
    elif loss_func_name == "1exp_loss":
        def loss_fun(C1, C2):
            return 1 - np.exp(np.abs(C1 - C2))
    elif loss_func_name == "1exp_loss_sigma":
        def loss_fun(C1, C2):
            return 1 - np.exp(np.abs(C1 - C2) / 10)
    elif loss_func_name == "inverse_exp_loss":
        def loss_fun(C1, C2):
            return 1 / np.exp(-(C1 - C2) ** 2) - 1
    elif loss_func_name == "square_root":
        def loss_fun(C1, C2):
            return (np.abs(C1 - C2)) ** 0.5
    else:
        print("Change the name of the loss")
        assert False
    return loss_fun


def find_best_T(C1, C2, loss_func, n_samples_s, n_samples_t, nb_perm=100):
    print("Trying to find the best T")
    assert n_samples_s == n_samples_t  # TODO can be improved
    best_T = 0
    best_loss = 100000
    identity = np.eye(n_samples_s)
    for i in range(nb_perm):
        print(".", end="")
        T = np.random.permutation(identity)
        best_actual_loss = compute_distance(T, C1, C2, loss_func)
        if best_loss > best_actual_loss:
            best_loss = best_actual_loss
            best_T = T
    print("")
    return best_T, best_loss


def solve_G_OT(name_algo, C1, C2, Xs, Xt, loss_func, args):
    T, W_distance, time_list_only_T = None, None, None
    time_before_run = time.time()

    if name_algo == "sampled_gromov_sliced":
        T = Generalisation_OT(C1=C1,
                              C2=C2,
                              loss_fun=loss_func,
                              T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
                              dimension_OT=args.dimension,
                              nb_iter_global=(0, 1),
                              epsilon_min=0,
                              repeat=args.repeat,
                              iter_epsilon=args.iter_epsilon,
                              nb_iter_batch=args.nb_iter_batch,
                              batch_size=args.batch_size,
                              constraint=args.constraint,
                              epsilon_init=0,
                              KL=0,
                              verbose=args.verbose,
                              sliced=True,
                              learning_step=args.learning_step)
        time_list_only_T = time.time() - time_before_run
        W_distance = Gromov.compute_distance_sampling_both(C1=C1,
                                                           C2=C2,
                                                           loss_fun=loss_func,
                                                           T=T)

    elif name_algo == "sampled_gromov_green":
        T = Generalisation_OT(C1=C1,
                              C2=C2,
                              loss_fun=loss_func,
                              T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
                              dimension_OT=args.dimension,
                              nb_iter_global=(0, 1),
                              epsilon_min=args.epsilon_min,
                              repeat=args.repeat,
                              iter_epsilon=args.iter_epsilon,
                              nb_iter_batch=args.nb_iter_batch,
                              batch_size=args.batch_size,
                              constraint=args.constraint,
                              epsilon_init=args.epsilon,
                              KL=args.KL,
                              verbose=args.verbose,
                              greenkhorn=True,
                              learning_step=args.learning_step)
        time_list_only_T = time.time() - time_before_run
        W_distance = Gromov.compute_distance_sampling_both(C1=C1,
                                                           C2=C2,
                                                           loss_fun=loss_func,
                                                           T=T)
    elif name_algo == "sampled_gromov_e_constant":
        T = Generalisation_OT(C1=C1,
                              C2=C2,
                              loss_fun=loss_func,
                              T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
                              dimension_OT=args.dimension,
                              nb_iter_global=(0, 1),
                              epsilon_min=args.epsilon,  # swap between the epsilon !
                              repeat=args.repeat,
                              iter_epsilon=args.iter_epsilon,
                              nb_iter_batch=args.nb_iter_batch,
                              batch_size=args.batch_size,
                              constraint=args.constraint,
                              epsilon_init=0,
                              KL=args.KL,
                              verbose=args.verbose,
                              learning_step=args.learning_step)
        time_list_only_T = time.time() - time_before_run
        W_distance = Gromov.compute_distance_sampling_both(C1=C1,
                                                           C2=C2,
                                                           loss_fun=loss_func,
                                                           T=T)

    elif name_algo == "sampled_gromov_no_KL_e_constant":
        T = Generalisation_OT(C1=C1,
                              C2=C2,
                              loss_fun=loss_func,
                              T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
                              dimension_OT=args.dimension,
                              nb_iter_global=(0, 1),
                              epsilon_min=args.epsilon,  # swap between the epsilon !
                              repeat=args.repeat,
                              iter_epsilon=args.iter_epsilon,
                              nb_iter_batch=args.nb_iter_batch,
                              batch_size=args.batch_size,
                              constraint=args.constraint,
                              epsilon_init=0,
                              KL=0,
                              verbose=args.verbose,
                              learning_step=args.learning_step)
        time_list_only_T = time.time() - time_before_run
        W_distance = Gromov.compute_distance_sampling_both(C1=C1,
                                                           C2=C2,
                                                           loss_fun=loss_func,
                                                           T=T)
#--------------- The three next algorithm has not been used -----------------
        # elif name_algo == "sampled_gromov":
        #     T = Generalisation_OT(C1=C1,
        #                           C2=C2,
        #                           loss_fun=loss_func,
        #                           T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
        #                           dimension_OT=args.dimension,
        #                           nb_iter_global=(0, 1),
        #                           epsilon_min=args.epsilon_min,
        #                           repeat=args.repeat,
        #                           iter_epsilon=args.iter_epsilon,
        #                           nb_iter_batch=args.nb_iter_batch,
        #                           batch_size=args.batch_size,
        #                           constraint=args.constraint,
        #                           epsilon_init=args.epsilon,
        #                           KL=args.KL,
        #                           verbose=args.verbose,
        #                           learning_step=args.learning_step)
        #     time_list_only_T = time.time() - time_before_run
        #     W_distance = Gromov.compute_distance_sampling_both(C1=C1,
        #                                                        C2=C2,
        #                                                        loss_fun=loss_func,
        #                                                        T=T)
        #
        # elif name_algo == "sampled_gromov_no_KL":
        #     T = Generalisation_OT(C1=C1,
        #                           C2=C2,
        #                           loss_fun=loss_func,
        #                           T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
        #                           dimension_OT=args.dimension,
        #                           nb_iter_global=(0, 1),
        #                           epsilon_min=args.epsilon_min,
        #                           repeat=args.repeat,
        #                           iter_epsilon=args.iter_epsilon,
        #                           nb_iter_batch=args.nb_iter_batch,
        #                           batch_size=args.batch_size,
        #                           constraint=args.constraint,
        #                           epsilon_init=args.epsilon,
        #                           KL=0,
        #                           verbose=args.verbose,
        #                           learning_step=args.learning_step)
        #     time_list_only_T = time.time() - time_before_run
        #
        #     W_distance = Gromov.compute_distance_sampling_both(C1=C1,
        #                                                        C2=C2,
        #                                                        loss_fun=loss_func,
        #                                                        T=T)

        # elif name_algo == "sampled_gromov_no_KL_e":
        #     T = Generalisation_OT(C1=C1,
        #                           C2=C2,
        #                           loss_fun=loss_func,
        #                           T=np.outer(ot.unif(args.n_samples_s), ot.unif(args.n_samples_t)),
        #                           dimension_OT=args.dimension,
        #                           nb_iter_global=(0, 1),
        #                           epsilon_min=0,
        #                           repeat=args.repeat,
        #                           iter_epsilon=args.iter_epsilon,
        #                           nb_iter_batch=args.nb_iter_batch,
        #                           batch_size=args.batch_size,
        #                           constraint=args.constraint,
        #                           epsilon_init=0,
        #                           KL=0,
        #                           verbose=args.verbose,
        #                           sliced=False,
        #                           learning_step=args.learning_step)
        #     time_list_only_T = time.time() - time_before_run
        #     W_distance = Gromov.compute_distance_sampling_both(C1=C1,
        #                                                        C2=C2,
        #                                                        loss_fun=loss_func,
        #                                                        T=T)


    #  ------------------Other methods-----------------------

    elif name_algo == "e_gromov":
        if args.loss_func_name in ["square_loss", "kl_loss"]:
            gromov_loss_func = args.loss_func_name
        else:
            gromov_loss_func = loss_func
        T, log = entropic_gromov_wasserstein(C1=C1,
                                             C2=C2,
                                             p=ot.unif(args.n_samples_s),
                                             q=ot.unif(args.n_samples_t),
                                             loss_fun=gromov_loss_func,
                                             epsilon=args.epsilon,
                                             log=True)
        time_list_only_T = time.time() - time_before_run
        # W_distance = Gromov.compute_distance(T, C1, C2, loss_func)
        W_distance = log["gw_dist"]

    elif name_algo == "e_gromov_KL":
        if args.loss_func_name in ["square_loss", "kl_loss"]:
            gromov_loss_func = args.loss_func_name
        else:
            gromov_loss_func = loss_func
        T, log = entropic_gromov_wasserstein(C1=C1,
                                             C2=C2,
                                             p=ot.unif(args.n_samples_s),
                                             q=ot.unif(args.n_samples_t),
                                             loss_fun=gromov_loss_func,
                                             epsilon=args.epsilon,
                                             KL=True,
                                             log=True)
        time_list_only_T = time.time() - time_before_run
        # W_distance = Gromov.compute_distance(T, C1, C2, loss_func)
        W_distance = log["gw_dist"]

    elif name_algo == "gromov":
        if args.loss_func_name in ["square_loss", "kl_loss"]:
            gromov_loss_func = args.loss_func_name
        else:
            gromov_loss_func = loss_func
        T, log = entropic_gromov_wasserstein(C1=C1,
                                             C2=C2,
                                             p=ot.unif(args.n_samples_s),
                                             q=ot.unif(args.n_samples_t),
                                             loss_fun=gromov_loss_func,
                                             epsilon=0,
                                             log=True)
        time_list_only_T = time.time() - time_before_run
        # if args.loss_func_name in ["square_loss", "kl_loss"]:
        #     constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)
        # else:
        # W_distance = Gromov.compute_distance(T, C1, C2, loss_func)
        W_distance = log["gw_dist"]
    elif name_algo == "sliced_gromov":
        assert Xs is not None  # Can't use a graph with sliced gromov
        assert Xt is not None
        if args.loss_func_name in ["square_loss", "kl_loss"]:
            # This is exactly the original algorithm.
            sliced_loss_func = args.loss_func_name
        else:
            # This is a modified algorithm that work with any loss function.
            sliced_loss_func = loss_func
        T, W_distance = sgw_cpu(xs=Xs, xt=Xt, loss_fun=sliced_loss_func, nproj=200)
        time_list_only_T = time.time() - time_before_run

    elif name_algo == "S_GWL":
        import warnings
        warnings.filterwarnings("ignore")
        if args.loss_func_name in ["square_loss", "kl_loss"]:
            # This is exactly the original algorithm.
            sgwl_loss_func = args.loss_func_name
        else:
            # This is a modified algorithm that work with any loss function.
            sgwl_loss_func = loss_func

        num_iters = 4000
        ot_dict = {'loss_type': sgwl_loss_func,  # the key hyperparameters of GW distance
                   'ot_method': 'proximal',
                   'beta': args.epsilon,
                   'outer_iteration': num_iters,
                   # outer, inner iteration, error bound of optimal transport
                   'iter_bound': 1e-30,
                   'inner_iteration': 2,
                   'sk_bound': 1e-30,
                   'node_prior': 1e3,
                   'max_iter': 4,  # iteration and error bound for calcuating barycenter
                   'cost_bound': 1e-26,
                   'update_p': False,  # optional updates of source distribution
                   'lr': 0,
                   'alpha': 0}

        idx2node_s = {}
        idx2node_t = {}
        for i in range(args.n_samples_s):
            idx2node_s[i] = i
        for i in range(args.n_samples_t):
            idx2node_t[i] = i

        pairs_idx, _, _ = GwGt.recursive_direct_graph_matching(
            cost_s=C1, cost_t=C2,
            p_s=ot.unif(args.n_samples_s)[:, np.newaxis],
            p_t=ot.unif(args.n_samples_t)[:, np.newaxis],
            idx2node_s=idx2node_s, idx2node_t=idx2node_t, ot_hyperpara=ot_dict,
            weights=None, predefine_barycenter=False, cluster_num=2,
            partition_level=3, max_node_num=0)

        pairs_idx = np.array(pairs_idx)
        T = np.zeros((C1.shape[0], C2.shape[0]))
        T[pairs_idx[:, 0], pairs_idx[:, 1]] = 1 / pairs_idx.shape[0]

        time_list_only_T = time.time() - time_before_run

        W_distance = compute_distance_sparse(C1, C2, loss_fun=loss_func,
                                             T=[pairs_idx[:, 0],
                                                pairs_idx[:, 1],
                                                np.ones(pairs_idx.shape[0]) / pairs_idx.shape[0]],
                                             dim_T=T.shape)

    elif name_algo == "uniform":
        T = np.ones((args.n_samples_s, args.n_samples_t)) / (args.n_samples_s * args.n_samples_t)
        time_list_only_T = time.time() - time_before_run
        # W_distance = compute_distance(C1=C1, C2=C2, loss=loss_func, T=T)
        if args.loss_func_name in ["square_loss", "kl_loss"]:
            constC, hC1, hC2 = Gromov.init_matrix(C1, C2,
                                                  np.ones(len(C1)) / len(C1),
                                                  np.ones(len(C2)) / len(C2),
                                                  args.loss_func_name)
            W_distance = Gromov.gwloss(constC, hC1, hC2, T)
        else:
            W_distance = Gromov.compute_distance(T, C1, C2, loss_func)
    elif name_algo == "identity":
        assert args.n_samples_s == args.n_samples_t
        T = [np.arange(args.n_samples_s), np.arange(args.n_samples_s), ot.unif(args.n_samples_s)]
        time_list_only_T = time.time() - time_before_run
        W_distance = compute_distance_sparse(C1=C1, C2=C2, loss_fun=loss_func,
                                             T=T,
                                             dim_T=[args.n_samples_s, args.n_samples_t])
    else:
        raise Exception("Name of method not found")
    time_list_only_W_approx = time.time() - time_before_run - time_list_only_T

    return T, W_distance, time_list_only_T, time_list_only_W_approx


def hyperparameter_analysis(n_samples_s=500,
                            name_dataset="gaussian",
                            pickle_path="pickle_compare",
                            names_algo=["sampled_gromov"],
                            names_algo_legend=["sampled_gromov"],
                            pickle_name="",
                            noise_graph=0.1,
                            loss_func_name="1_loss",
                            figsize=(10, 5),
                            save=False,
                            entropy=[0.1],
                            batchsize=[1],
                            iteration=[1000],
                            log_scale=False,
                            min_plot=False,
                            only_PGW=True,
                            final_plot=True):
    plt.figure(0, figsize=figsize)
    already_plot_SGW_legend = False
    if name_dataset in ["gaussian_graph", "same_graph", "gaussian_point_graph"]:
        # Can't compare sliced with other.
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("Sliced GW")
        except:
            pass
    params = {}
    same_param = None
    markersize = 10
    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        if name_algo in ["e_gromov", "e_gromov_KL"]:  # "e_gromov_KL"
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    label = None
                    if name_algo == "e_gromov":
                        marker = 'o'
                    else:
                        marker = 'P'
                    if e_index == 0:
                        if name_algo == "e_gromov":
                            label = "$EGW$"
                            plt.plot([], [],
                                     marker=marker,
                                     color=(0, 0, 0),
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                        else:
                            label = "$EGW^{KL}$"
                            plt.plot([], [],
                                     marker=marker,
                                     color=(0, 0, 0),
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                    markersize = 10
                    label = None
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)

                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            # acces_dict = "gromov" + str(n_samples_s) + "_" + str(n_samples_s)
            # if params[acces_dict] is not None:
            #     m_save_time = [np.mean(params[acces_dict]["time_list_only_T"])] + m_save_time
            #     m_save_distance = [np.mean(params[acces_dict]["W_distance_list"])] + m_save_distance
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=(0, 0.5, 0.5),
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["S_GWL"]:
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))

                    marker = "*"
                    # print(e_index, e)
                    if not already_plot_SGW_legend and final_plot:
                        already_plot_SGW_legend = True
                        plt.plot([], [],
                                 marker=marker,
                                 color=(0, 0, 0),
                                 label=name_algo_legend,
                                 markersize=markersize,
                                 linestyle="None")
                    # markersize = 10
                    label = None
                    # if not final_plot:
                    #     label = name_algo_legend
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)

                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            # acces_dict = "gromov" + str(n_samples_s) + "_" + str(n_samples_s)
            # if params[acces_dict] is not None:
            #     m_save_time = [np.mean(params[acces_dict]["time_list_only_T"])] + m_save_time
            #     m_save_distance = [np.mean(params[acces_dict]["W_distance_list"])] + m_save_distance
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=(0, 0.5, 0.5),
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            if name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant":
                plt.plot([], [],
                         marker="",
                         color=(0, 0, 0),
                         label="$SaGroW^{KL}$" if name_algo == "sampled_gromov_e_constant" else "$SaGroW$",
                         markersize=2,
                         linestyle="None")
                plt.plot([], [],
                         marker="X",
                         color=(0, 0, 0),
                         label="PoGroW",
                         markersize=10,
                         linestyle="None")
                plt.plot([], [],
                         marker="x",
                         color=(0, 0, 0),
                         label="Minimum\nover\n10 runs",
                         markersize=10,
                         linestyle="None")
                if final_plot:
                    plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                    if iteration[-1] == 1000:  # and loss_func_name == "square_loss":
                        plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )

            for b_index, b in enumerate(batchsize):
                m_save_time = []
                m_save_distance = []
                for m_index, m in enumerate(iteration):

                    acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(b) + str(m)

                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(
                            noise_graph) + "_" + \
                                      str(b) + "_" + str(m) + "_" + pickle_name + ".pickle"
                    elif name_algo == "sampled_gromov_no_KL_e_constant":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + "_3" + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + pickle_name + ".pickle"

                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:
                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                        time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                        time_list_only_T = params[acces_dict]["time_list_only_T"]
                        time_list_only_W = params[acces_dict]["time_list_only_W"]
                        time_list = time_list_only_T + time_list_only_W
                        W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                        W_distance_list = params[acces_dict]["W_distance_list"]

                        label = None
                        markersize = 10

                        if b == 1 and name_algo == "sampled_gromov_sliced" and only_PGW:
                            markersize = 10
                            marker = "X"
                            # print(1)
                            # print((1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3)
                            color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            # color = (1, (1 - m_index / (len(iteration) - 1)) / 1.2, 0)
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)

                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=None,
                                     markersize=markersize,
                                     linestyle="None")
                            plt.plot([np.mean(time_list_only_T), np.sum(time_list_only_T)],
                                     [np.mean(W_distance_list), np.min(W_distance_list)],
                                     color=color,
                                     marker=None,
                                     linestyle="dotted")
                            plt.plot(np.sum(time_list_only_T), np.min(W_distance_list),
                                     color=color,
                                     marker="x",
                                     linestyle=None)

                        else:
                            markersize = 10
                            marker = (b_index + 2, 0, 0)
                            color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)

                            # color = 1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                            if name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant":
                                # if plot:
                                if m_index == 0:
                                    plt.plot([None], [None],
                                             marker=marker,
                                             color=(0, 0, 0),
                                             label=str(b) + " sample" + ((b != 1) * "s"),
                                             markersize=markersize,
                                             linestyle="None")
                                if b_index == (len(batchsize) - 1):
                                    if iteration[-1] == 1000 and m_index == 0:  # and loss_func_name == "loss_square":
                                        plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                                    plt.plot([None], [None],
                                             marker="s",  # (3, 0, 0),
                                             color=color,
                                             label=str(m) + " iterations",
                                             markersize=8, )
                                    # linestyle="None")

                                # if b_index == 2 and m_index == 2:
                                #     label = "$SGW^{KL}$"
                            else:
                                if only_PGW:
                                    continue

                                    # marker = (b_index + 2, 0, 0)
                                    # color = (1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                                    #
                                    # if m_index == 0:
                                    #     plt.plot([None], [None],
                                    #              marker=marker,
                                    #              color=(0, 0, 0),
                                    #              label=str(b) + " sample" + ((b != 1) * "s"),
                                    #              markersize=markersize,
                                    #              linestyle="None")
                                    # if b_index == (len(batchsize) - 1):
                                    #     plt.plot([None], [None],
                                    #              marker=(3, 0, 0),
                                    #              color=color,
                                    #              label=str(m) + " iterations",
                                    #              markersize=markersize,
                                    #              linestyle="None")
                            #     marker = (b_index + 2, 0, 0)
                            #     color = (0, 0, 1 - m_index / (len(iteration) - 1))
                            #     if b_index == 2 and m_index == 2:
                            #         label = "$SGW$"
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)
                            m_save_time.append(time_list_only_T_)
                            m_save_distance.append(W_distance_list_)
                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                    else:
                        params[acces_dict] = None
                plt.plot(m_save_time, m_save_distance,
                         "-",
                         color=(1, 0.5, 0),
                         markersize=3,
                         linewidth=0.5)

        else:
            acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s)

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:

                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
                time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                time_list_only_T = params[acces_dict]["time_list_only_T"]
                time_list_only_W = params[acces_dict]["time_list_only_W"]
                time_list = time_list_only_T + time_list_only_W
                W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                W_distance_list = params[acces_dict]["W_distance_list"]
                label = name_algo_legend
                # if name_algo_legend[0] == "$":
                #     marker = name_algo_legend
                # else:
                #     marker = "$" + name_algo_legend + "$"
                # markersize = 50
                # color = (0.5,1,0.5)

                if name_algo == "S_GWL":
                    color = (0, 0, 0)
                    marker = "+"
                elif name_algo == "e_gromov_KL":
                    color = (0, 0, 1)
                elif name_algo == "gromov":
                    color = (0, 1, 0)
                    marker = "+"
                elif name_algo == "sampled_gromov":
                    color = (0, 1, 0.5)
                    marker = "+"

                if not name_algo in ["uniform", "identity"]:
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    plt.plot(np.mean(time_list_only_T_), np.mean(W_distance_list_),
                             marker=marker,
                             label=label,
                             markersize=markersize,
                             linestyle="None",
                             color=color)
                else:
                    label = name_algo_legend
                    if name_algo == "uniform":
                        # color = (1, 1/1.4, 0)
                        color = (219. / 255., 112. / 255., 147. / 255.)
                        entropy_string = ["Low $\epsilon$", "Medium $\epsilon$", "High $\epsilon$"]
                        entropy_color = [(0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
                        for ent in range(3):
                            plt.plot([], [],
                                     marker="s",
                                     label=entropy_string[ent],
                                     color=entropy_color[ent],
                                     markersize=8)
                    else:
                        color = (1, 0, 0)
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    plt.axhline(y=W_distance_list_, color=color, linestyle="-", label=label)
                    # plt.plot([1e-2, 1e4], [W_distance_list_] * 2,
                    #          label=label,
                    #          markersize=markersize,
                    #          color=color)
            else:
                # print("outer")
                params[acces_dict] = None

    plt.xscale("log")

    if final_plot:
        plt.xlabel("Computational time (s)", fontsize=16)
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)", fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        from matplotlib.legend_handler import HandlerLine2D
        import matplotlib.lines

        class SymHandler(HandlerLine2D):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                xx = 0.6 * height
                return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height,
                                                              fontsize, trans)

        plt.legend(handler_map={matplotlib.lines.Line2D: SymHandler()},
                   bbox_to_anchor=(-0.12, 0.9, 1.2, 0.3), ncol=5, handleheight=2, labelspacing=0,
                   # bbox_to_anchor=(-0.12, 0.95, 1.25, 0.3), ncol=5, handleheight=2, labelspacing=0,
                   prop={'size': 13.5},
                   frameon=False)
    else:
        plt.xlabel("Computational time (s)")
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)")
        # plt.xticks(fontsize=16)
        plt.legend()
    if save:
        if "sampled_gromov_no_KL_e_constant" in names_algo:
            plt.savefig(
                "./figure/Analysis_hyper/noKL_" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + ".pdf", bbox_inches="tight")
        else:
            plt.savefig(
                "./figure/Analysis_hyper/" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + ".pdf", bbox_inches="tight")
    plt.show()


def hyperparameter_analysis_3(n_samples_s=500,
                              name_dataset="gaussian",
                              pickle_path="pickle_compare",
                              names_algo=["sampled_gromov"],
                              names_algo_legend=["sampled_gromov"],
                              pickle_name="",
                              noise_graph=0.1,
                              loss_func_name="1_loss",
                              figsize=(10, 5),
                              save=False,
                              entropy=[0.1],
                              batchsize=[1],
                              iteration=[1000],
                              log_scale=False,
                              min_plot=False,
                              only_PGW=True,
                              final_plot=True,
                              which_plot=0):
    grey_color = (0.9, 0.9, 0.9)
    fig = plt.figure(0, figsize=figsize)
    already_plot_SGW_legend = False
    if name_dataset in ["gaussian_graph", "same_graph", "gaussian_point_graph"]:
        # Can't compare sliced with other.
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("Sliced GW")
        except:
            pass
    params = {}
    same_param = None
    markersize = 10
    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        if name_algo in ["e_gromov", "e_gromov_KL"]:  # "e_gromov_KL"
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    # print(name_algo, color, entropy[e_index])
                    label = None
                    if name_algo == "e_gromov":
                        marker = 'o'
                    else:
                        marker = 'P'
                    if e_index == 0:
                        if name_algo == "e_gromov":
                            label = "$EGW$"
                            if which_plot == 2:
                                plt.plot([], [],
                                         marker=marker,
                                         color=(0, 0, 0),
                                         label=label,
                                         markersize=markersize,
                                         linestyle="None")
                        else:
                            label = "$EGW^{KL}$"
                            if which_plot == 2:
                                plt.plot([], [],
                                         marker=marker,
                                         color=(0, 0, 0),
                                         label=label,
                                         markersize=markersize,
                                         linestyle="None")
                    markersize = 10
                    label = None
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        # print("std", np.std(W_distance_list))
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)
                    if which_plot != 2:
                        color = grey_color
                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            if which_plot != 2:
                color = grey_color
            else:
                color = (0, 0.5, 0.5)
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=color,
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["S_GWL"]:
            m_save_time = []
            m_save_distance = []
            for e_index, e in enumerate(entropy):
                acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(e)

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  str(e) + "__" + pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s) + "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(
                        e) + "_" + pickle_name + ".pickle"
                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:
                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                    time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T = params[acces_dict]["time_list_only_T"]
                    time_list_only_W = params[acces_dict]["time_list_only_W"]
                    time_list = time_list_only_T + time_list_only_W
                    W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                    W_distance_list = params[acces_dict]["W_distance_list"]
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))

                    marker = "*"
                    # print(e_index, e)
                    if not already_plot_SGW_legend and final_plot:
                        already_plot_SGW_legend = True
                        if which_plot == 2:
                            plt.plot([], [],
                                     marker=marker,
                                     color=(0, 0, 0),
                                     label=name_algo_legend,
                                     markersize=markersize,
                                     linestyle="None")
                    # markersize = 10
                    label = None
                    # if not final_plot:
                    #     label = name_algo_legend
                    color = (0, 1 - e_index / (len(entropy) - 1), e_index / (len(entropy) - 1))
                    # print("SGWL", color, entropy[e_index])

                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        # print(np.std(W_distance_list))
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    m_save_time.append(time_list_only_T_)
                    m_save_distance.append(W_distance_list_)
                    if which_plot != 2:
                        color = grey_color
                    plt.plot(time_list_only_T_, W_distance_list_,
                             marker=marker,
                             color=color,
                             label=label,
                             markersize=markersize,
                             linestyle="None")
                else:
                    params[acces_dict] = None
            # acces_dict = "gromov" + str(n_samples_s) + "_" + str(n_samples_s)
            # if params[acces_dict] is not None:
            #     m_save_time = [np.mean(params[acces_dict]["time_list_only_T"])] + m_save_time
            #     m_save_distance = [np.mean(params[acces_dict]["W_distance_list"])] + m_save_distance
            if which_plot != 2:
                color = grey_color
            else:
                color = (0, 0.5, 0.5)
            plt.plot(m_save_time,
                     m_save_distance,
                     "-",
                     color=color,
                     markersize=3,
                     linewidth=0.5)

        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            if (name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant"):
                if which_plot == 1:
                    plt.plot([], [],
                             marker="",
                             color=(0, 0, 0),
                             label="$SaGroW^{KL}$" if name_algo == "sampled_gromov_e_constant" else "$SaGroW$",
                             markersize=2,
                             linestyle="None")
                if which_plot == 0:
                    plt.plot([], [],
                             marker="X",
                             color=(0, 0, 0),
                             label="PoGroW",
                             markersize=10,
                             linestyle="None")
                    plt.plot([], [],
                             marker="x",
                             color=(0, 0, 0),
                             label="Minimum over 10 runs",
                             markersize=10,
                             linestyle="None")
                # if final_plot:
                #     plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                #     if iteration[-1] == 1000:  # and loss_func_name == "square_loss":
                #         plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )

            for b_index, b in enumerate(batchsize):
                m_save_time = []
                m_save_distance = []
                for m_index, m in enumerate(iteration):

                    acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s) + str(b) + str(m)

                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(
                            noise_graph) + "_" + \
                                      str(b) + "_" + str(m) + "_" + pickle_name + ".pickle"
                    elif name_algo == "sampled_gromov_no_KL_e_constant" and which_plot == 1:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + "_3" + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s) + \
                                      "_" + str(n_samples_s) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                            m) + pickle_name + ".pickle"

                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:
                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                        time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                        time_list_only_T = params[acces_dict]["time_list_only_T"]
                        time_list_only_W = params[acces_dict]["time_list_only_W"]
                        time_list = time_list_only_T + time_list_only_W
                        W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                        W_distance_list = params[acces_dict]["W_distance_list"]

                        label = None
                        markersize = 10

                        if b == 1 and name_algo == "sampled_gromov_sliced" and only_PGW:
                            markersize = 10
                            marker = "X"
                            # print(1)
                            # print((1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3)
                            if which_plot == 0:
                                color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            else:
                                color = grey_color
                            # color = (1, (1 - m_index / (len(iteration) - 1)) / 1.2, 0)
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)

                            if which_plot == 0:
                                alpha = 0.15
                                plt.vlines(time_list_only_T_,
                                                 W_distance_list_ - np.std(W_distance_list),
                                                 W_distance_list_ + np.std(W_distance_list),
                                                 alpha=alpha,
                                                 # marker="_",
                                                 color=color,)
                                plt.plot([time_list_only_T_, time_list_only_T_],
                                         [W_distance_list_ + np.std(W_distance_list),
                                          W_distance_list_ - np.std(W_distance_list)],
                                         alpha=alpha,
                                         color=color,
                                         marker="_")
                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=None,
                                     markersize=markersize,
                                     linestyle="None")
                            plt.plot([np.mean(time_list_only_T), np.sum(time_list_only_T)],
                                     [np.mean(W_distance_list), np.min(W_distance_list)],
                                     color=color,
                                     marker=None,
                                     linestyle="dotted")
                            plt.plot(np.sum(time_list_only_T), np.min(W_distance_list),
                                     color=color,
                                     marker="x",
                                     linestyle=None)


                        else:
                            markersize = 10
                            marker = (b_index + 3, 0, 0)
                            # if which_plot == 1:
                            color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            # else:
                            #     color = grey_color
                            # color = (1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                            if (
                                    name_algo == "sampled_gromov_e_constant" or name_algo == "sampled_gromov_no_KL_e_constant"):
                                if m_index == 0 and which_plot == 1:
                                    plt.plot([None], [None],
                                             marker=marker,
                                             color=(0, 0, 0),
                                             label=str(b) + " sample" + ((b != 1) * "s"),
                                             markersize=markersize,
                                             linestyle="None")
                                if b_index == (len(batchsize) - 1) and which_plot < 2:
                                    if iteration[-1] == 1000 and m_index == 0:  # and loss_func_name == "loss_square":
                                        plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0), )
                                    plt.plot([None], [None],
                                             marker="s",  # (3, 0, 0),
                                             color=color,
                                             label=str(m) + " iterations",
                                             markersize=8)

                            else:
                                if only_PGW:
                                    continue

                                    # marker = (b_index + 2, 0, 0)
                                    # color = (1, ((1 - m_index / (len(iteration) - 1))) / 1.5, 0)
                                    #
                                    # if m_index == 0:
                                    #     plt.plot([None], [None],
                                    #              marker=marker,
                                    #              color=(0, 0, 0),
                                    #              label=str(b) + " sample" + ((b != 1) * "s"),
                                    #              markersize=markersize,
                                    #              linestyle="None")
                                    # if b_index == (len(batchsize) - 1):
                                    #     plt.plot([None], [None],
                                    #              marker=(3, 0, 0),
                                    #              color=color,
                                    #              label=str(m) + " iterations",
                                    #              markersize=markersize,
                                    #              linestyle="None")
                            #     marker = (b_index + 2, 0, 0)
                            #     color = (0, 0, 1 - m_index / (len(iteration) - 1))
                            #     if b_index == 2 and m_index == 2:
                            #         label = "$SGW$"
                            if min_plot:
                                W_distance_list_ = np.min(W_distance_list)
                                time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                            else:
                                W_distance_list_ = np.mean(W_distance_list)
                                time_list_only_T_ = np.mean(time_list_only_T)
                            m_save_time.append(time_list_only_T_)
                            m_save_distance.append(W_distance_list_)
                            if which_plot == 1:
                                color = (1, (1 - (m_index / (len(iteration) - 1))) ** 0.45 / 1.3, 0)
                            else:
                                color = grey_color
                            if which_plot == 1:
                                alpha = 0.15
                                plt.vlines(time_list_only_T_,
                                                 W_distance_list_ - np.std(W_distance_list),
                                                 W_distance_list_ + np.std(W_distance_list),
                                                 alpha=alpha,
                                                 # marker="_",
                                                 color=color,)
                                plt.plot([time_list_only_T_, time_list_only_T_],
                                         [W_distance_list_ + np.std(W_distance_list),
                                          W_distance_list_ - np.std(W_distance_list)],
                                         alpha=alpha,
                                         color=color,
                                         marker="_")
                            plt.plot(time_list_only_T_, W_distance_list_,
                                     marker=marker,
                                     color=color,
                                     label=label,
                                     markersize=markersize,
                                     linestyle="None")
                    else:
                        params[acces_dict] = None
                if which_plot == 1:
                    color = (1, 0.5, 0)
                else:
                    color = grey_color

                plt.plot(m_save_time, m_save_distance,
                         "-",
                         color=color,
                         markersize=3,
                         linewidth=0.5)

        elif name_algo in ["e_gromov_KL", "e_gromov", "gromov", "S_GWL"]:
            acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s)

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:

                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
                time_list_only_W_approx = params[acces_dict]["time_list_only_W_approx"]
                time_list_only_T = params[acces_dict]["time_list_only_T"]
                time_list_only_W = params[acces_dict]["time_list_only_W"]
                time_list = time_list_only_T + time_list_only_W
                W_distance_list_approx = params[acces_dict]["W_distance_list_approx"]
                W_distance_list = params[acces_dict]["W_distance_list"]
                label = name_algo_legend
                # if name_algo_legend[0] == "$":
                #     marker = name_algo_legend
                # else:
                #     marker = "$" + name_algo_legend + "$"
                # markersize = 50
                # color = (0.5,1,0.5)

                if name_algo == "S_GWL":
                    color = (0, 0, 0)
                    marker = "+"
                elif name_algo == "e_gromov_KL":
                    color = (0, 0, 1)
                elif name_algo == "gromov":
                    if which_plot == 2:
                        color = (0, 1, 0)
                    else:
                        color = grey_color
                        label = ""
                    marker = "+"
                elif name_algo == "sampled_gromov":
                    color = (0, 1, 0.5)
                    marker = "+"
                # elif name_algo == "uniform":
                #     marker = ""

                if not name_algo in ["uniform", "identity"]:
                    if min_plot:
                        W_distance_list_ = np.min(W_distance_list)
                        time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                    else:
                        W_distance_list_ = np.mean(W_distance_list)
                        time_list_only_T_ = np.mean(time_list_only_T)
                    plt.plot(np.mean(time_list_only_T_), np.mean(W_distance_list_),
                             marker=marker,
                             label=label,
                             markersize=markersize,
                             linestyle="None",
                             color=color)
                # if name_algo == "gromov":
                #     # color = (1, 1/1.4, 0)
                #     color = (219. / 255., 112. / 255., 147. / 255.)
                #     entropy_string = ["Low $\epsilon$", "Medium $\epsilon$", "High $\epsilon$"]
                #     entropy_color = [(0, 1, 0), (0, 0.5, 0.5), (0, 0, 1)]
                #     for ent in range(3):
                #         plt.plot([], [],
                #                  marker="s",
                #                  label=entropy_string[ent],
                #                  color=entropy_color[ent],
                #                  markersize=8)
                # else:
                #     label = name_algo_legend
                #
                #     else:
                #         color = (1, 0, 0)
                #     if min_plot:
                #         W_distance_list_ = np.min(W_distance_list)
                #         time_list_only_T_ = time_list_only_T[np.argmin(W_distance_list)]
                #     else:
                #         W_distance_list_ = np.mean(W_distance_list)
                #         time_list_only_T_ = np.mean(time_list_only_T)
                #     plt.axhline(y=W_distance_list_, color=color, linestyle="-", label=label)
                # plt.plot([1e-2, 1e4], [W_distance_list_] * 2,
                #          label=label,
                #          markersize=markersize,
                #          color=color)
            else:
                # print("outer")
                params[acces_dict] = None

        elif name_algo in ["uniform"]:
            color = (0.9, 0.5, 0.9)
            acces_dict = name_algo + str(n_samples_s) + "_" + str(n_samples_s)

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                    n_samples_s) + \
                              "_" + str(n_samples_s) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:
                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
                time_list_only_T = params[acces_dict]["time_list_only_T"]
                time_list_only_W = params[acces_dict]["time_list_only_W"]
                W_distance_list = params[acces_dict]["W_distance_list"]
                label = name_algo_legend
                W_distance_list_ = np.mean(W_distance_list)
                time_list_only_T_ = np.mean(time_list_only_T)
                plt.axhline(y=W_distance_list_, color=color, linestyle="-", label=label)

    plt.xscale("log")

    if final_plot:
        plt.xlabel("Computational time (s)", fontsize=16)
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)", fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        from matplotlib.legend_handler import HandlerLine2D
        import matplotlib.lines

        class SymHandler(HandlerLine2D):
            def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
                xx = 0.6 * height
                return super(SymHandler, self).create_artists(legend, orig_handle, xdescent, xx, width, height,
                                                              fontsize, trans)

        # if which_plot == 0:

        handles, labels = plt.gca().get_legend_handles_labels()
        if which_plot == 0:
            bbox_to_anchor = (-0.3, 0.9, 1.3, 0.3)  # (-0.12, 0.9, 1.2, 0.3)
            plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0))
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 2, 1, 3, 6, 4, 7, 5]
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       handler_map={matplotlib.lines.Line2D: SymHandler()},
                       bbox_to_anchor=bbox_to_anchor, ncol=4, handleheight=2, labelspacing=0,
                       prop={'size': 12.5},
                       frameon=False)
        elif which_plot == 1:
            bbox_to_anchor = (-0.19, 0.9, 1.2, 0.3)  # (-0.12, 0.9, 1.2, 0.3)
            plt.plot([], [], label=" ", linestyle="None", marker="", color=(0, 0, 0))
            handles, labels = plt.gca().get_legend_handles_labels()
            order = [0, 9, 1, 4, 2, 5, 3, 6, 8, 7] #np.arange(len(labels))#
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       handler_map={matplotlib.lines.Line2D: SymHandler()},
                       bbox_to_anchor=bbox_to_anchor, ncol=5, handleheight=2, labelspacing=0,
                       columnspacing=1.5,
                       prop={'size': 12.5},
                       frameon=False)
        elif which_plot == 2:
            bbox_to_anchor = (-0.40, 0.9, 1.2, 0.3)  # (-0.12, 0.9, 1.2, 0.3)
            bbox_to_anchor = (21, 1.1, 1.4, 0.2)  # (-0.12, 0.9, 1.2, 0.3)
            order = [5, 1, 6, 2, 4, 3, 0, 7]
            order = [2, 3, 1, 0, 4]
            import matplotlib.colors as mcolors
            import matplotlib as mpl

            def make_colormap(seq):
                """Return a LinearSegmentedColormap
                seq: a sequence of floats and RGB-tuples. The floats should be increasing
                and in the interval (0,1).
                """
                seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
                # print(seq)
                cdict = {'red': [], 'green': [], 'blue': []}
                for i, item in enumerate(seq):
                    if isinstance(item, float):
                        r1, g1, b1 = seq[i - 1]
                        r2, g2, b2 = seq[i + 1]
                        cdict['red'].append([item, r1, r2])
                        cdict['green'].append([item, g1, g2])
                        cdict['blue'].append([item, b1, b2])
                return mcolors.LinearSegmentedColormap('CustomMap', cdict)

            # c = mcolors.ColorConverter().to_rgb
            rvb = make_colormap(
                # [(0, 1, 0), (0, 0.5, 0.5), float(10**(-3)), (0, 0.5, 0.5), (0, 0, 1), float(10**2), (0, 0, 1), 10**2])
            [(0, 1, 0), (0, 0.5, 0.5), 0.33, (0, 0.5, 0.5), (0, 0, 1), 0.66, (0, 0, 1)])
            norm = mpl.colors.LogNorm(vmin=0.001, vmax=100)
            # norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=rvb, norm=norm)

            sm._A = []
            # cbaxes = plt.add_axes
            cbaxes = fig.add_axes([0.15, 0.2, 0.03, 0.6])
            # , shrink = 0.8, anchor = (0, 0.5)
            cbar = plt.colorbar(sm, cax=cbaxes)#, pad=-0.5)#, norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))
            cbar.set_label('Epsilon : $\epsilon$', rotation=90, labelpad=-70)
            # cbar.set_clim(-2.0, 2.0)
            # plt.clim(-4, 4)
            plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                       handler_map={matplotlib.lines.Line2D: SymHandler()},
                       bbox_to_anchor=bbox_to_anchor, ncol=5, handleheight=2, labelspacing=0,
                       prop={'size': 12.5},
                       frameon=False)
        # else:
        #     order = np.arange(len(labels))




    else:
        plt.xlabel("Computational time (s)")
        plt.ylabel("Gromov Wasserstein value: $\mathcal{E}$ (T)")
        # plt.xticks(fontsize=16)
        plt.legend()
    if save:
        if "sampled_gromov_no_KL_e_constant" in names_algo:
            plt.savefig(
                "./figure/Analysis_hyper/noKL_" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + "_" + str(which_plot) + ".pdf", bbox_inches="tight")
        else:
            plt.savefig(
                "./figure/Analysis_hyper/" + str(min_plot) + "_" + loss_func_name + "_" + str(n_samples_s) + "_" +
                name_dataset + "_" + str(which_plot) + ".pdf", bbox_inches="tight")
    plt.show()


def analyse_data_hyper(name_dataset="gaussian",
                       pickle_path="pickle_compare",
                       names_algo=["sampled_gromov"],
                       names_algo_legend=["sampled_gromov"],
                       pickle_name="",
                       n_samples_s=[50, 100, 200],
                       n_samples_t=[],
                       noise_graph=0.1,
                       loss_func_name="1_loss",
                       best_T_not_needed=True,
                       log_scale=True,
                       figsize=(10, 5),
                       save=False,
                       entropy=[0.1],
                       entropy_SGWL=[0.1],
                       batchsize=[1],
                       iteration=[1000],
                       paper_plot=False,
                       show_legend=True,
                       legend_SGW=False,
                       markersize=10,
                       linestyle="dotted"
                       ):
    n_samples_s = np.array(n_samples_s)
    if name_dataset in ["gaussian_graph", "same_graph"]:
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("SGW")
        except:
            pass

    if name_dataset in ["gaussian_graph", "gaussian_point_graph"]:
        try:
            names_algo.remove("identity")
            names_algo_legend.remove("Identity")
        except:
            pass

    fontsize_legend = 17
    fontsize_axis = 17
    fontsize_ticks = 14

    if not n_samples_t:
        n_samples_t = n_samples_s

    if paper_plot:
        # I found manually the best param for each dataset.
        # Print the average best value over all the iteration, this give the best parameter.
        if name_dataset == "same_graph":
            entropy = [0.005]
        elif name_dataset == "gaussian_graph":
            entropy = [0.01]
        elif name_dataset == "gaussian_point_graph":
            entropy = [0.005]

        if name_dataset == "same_graph":
            entropy_SGWL = [100]
        elif name_dataset == "gaussian_graph":
            entropy_SGWL = [1]
        elif name_dataset == "gaussian_point_graph":
            entropy_SGWL = [10]

        if name_dataset == "same_graph":
            best_param_m = 500
            best_param_b = 10
        elif name_dataset == "gaussian_graph":
            best_param_m = 500
            best_param_b = 10
        elif name_dataset == "gaussian_point_graph":
            best_param_m = 100
            best_param_b = 10

    params = {}

    same_param = None

    for name_algo in names_algo:
        for i in range(len(n_samples_s)):
            if name_algo in ["e_gromov", "e_gromov_KL"]:
                for e in entropy:
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                      str(e) + "__" + pickle_name + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(
                            e) + "_" + pickle_name + ".pickle"
                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:

                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                    else:
                        # print(path_pickle)
                        # print("outer")
                        params[acces_dict] = None
            elif name_algo in ["S_GWL"]:
                for e in entropy_SGWL:
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if name_dataset == "same_graph":
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                      str(e) + "__" + pickle_name + ".pickle"
                    else:
                        path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                            n_samples_s[i]) + \
                                      "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(
                            e) + "_" + pickle_name + ".pickle"
                    if path.exists(path_pickle):
                        with open(path_pickle, "rb") as pickle_in:

                            params[acces_dict] = pickle.load(pickle_in)
                            if same_param is None:
                                same_param = params[acces_dict]
                            elif params[acces_dict]["name_algo"] != "uniform":
                                # Use this to check if every run use the same parameter
                                for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                    if same_param[test] != params[acces_dict][test]:
                                        print(params[acces_dict])
                                        pass
                                        print("Warning", test, same_param[test], params[acces_dict][test])
                                    # assert same_param[test] == params[acces_dict][test]
                    else:
                        # print(path_pickle)
                        # print("outer")
                        params[acces_dict] = None
            elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
                for b in batchsize:
                    for m in iteration:
                        acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)

                        if name_dataset == "same_graph":
                            path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                                n_samples_s[i]) + \
                                          "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(
                                noise_graph) + "_" + \
                                          str(b) + "_" + str(m) + "_" + pickle_name + ".pickle"
                        elif name_algo == "sampled_gromov_no_KL_e_constant":
                            path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                                n_samples_s[i]) + \
                                          "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                                m) + "_3" + ".pickle"
                        else:
                            path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                                n_samples_s[i]) + \
                                          "_" + str(n_samples_t[i]) + "_" + loss_func_name + "_" + str(b) + "_" + str(
                                m) + pickle_name + ".pickle"
                        if path.exists(path_pickle):
                            with open(path_pickle, "rb") as pickle_in:

                                params[acces_dict] = pickle.load(pickle_in)
                                if same_param is None:
                                    same_param = params[acces_dict]
                                elif params[acces_dict]["name_algo"] != "uniform":
                                    # Use this to check if every run use the same parameter
                                    for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                        if same_param[test] != params[acces_dict][test]:
                                            print(params[acces_dict])
                                            pass
                                            print("Warning", test, same_param[test], params[acces_dict][test])
                                        # assert same_param[test] == params[acces_dict][test]
                        else:
                            # print("outer")
                            params[acces_dict] = None
            else:
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])

                if name_dataset == "same_graph":
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s[i]) + \
                                  "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                                  pickle_name + ".pickle"
                else:
                    path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(
                        n_samples_s[i]) + \
                                  "_" + str(n_samples_t[i]) + "_" + loss_func_name + pickle_name + ".pickle"

                if path.exists(path_pickle):
                    with open(path_pickle, "rb") as pickle_in:

                        params[acces_dict] = pickle.load(pickle_in)
                        if same_param is None:
                            same_param = params[acces_dict]
                        elif params[acces_dict]["name_algo"] != "uniform":
                            # Use this to check if every run use the same parameter
                            for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                                if same_param[test] != params[acces_dict][test]:
                                    print(params[acces_dict])
                                    pass
                                    print("Warning", test, same_param[test], params[acces_dict][test])
                                # assert same_param[test] == params[acces_dict][test]
                else:
                    # print("outer")
                    params[acces_dict] = None

    assert same_param is not None  # Maybe an error in the pickle file name

    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        if name_algo in ["e_gromov", "e_gromov_KL"]:
            for e in entropy:
                time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                n_samples_s_temp = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                # n_samples_s_temp = n_samples_s_temp.astype(float)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    # print(params[acces_dict])
                    if params[acces_dict] is None:
                        # n_samples_s_temp[i] = np.NaN
                        mask[i] = 0
                        continue
                    time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                    time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                    time_list[i] = time_list_only_T[i] + time_list_only_W[i]
                # linestyle = "--"
                plt.figure(0, figsize=figsize)
                if name_algo == "e_gromov":
                    color = (0, 0.5, 0.8)
                    marker = "o"
                if name_algo == "e_gromov_KL":
                    color = (0, 0, 1)#(0, 0.8, 0.5)
                    marker = "P"
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                                 (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                                 alpha=0.3, color=color)

                plt.figure(1, figsize=figsize)

                plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=10)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                                 (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                                 alpha=0.3, color=color)
        elif name_algo in ["S_GWL"]:
            for e in entropy_SGWL:
                time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                n_samples_s_temp = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                # n_samples_s_temp = n_samples_s_temp.astype(float)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    # print(params[acces_dict])
                    if params[acces_dict] is None:
                        # n_samples_s_temp[i] = np.NaN
                        mask[i] = 0
                        continue
                    time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                    time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                    time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                    time_list[i] = time_list_only_T[i] + time_list_only_W[i]
                # linestyle = "--"
                plt.figure(0, figsize=figsize)
                color = (0, 0, 0)
                marker = "*"
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=10)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                                 (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                                 alpha=0.3, color=color)

                plt.figure(1, figsize=figsize)
                plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                         label=name_algo_legend + legend_param, color=color, marker=marker, markersize=10)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                                 (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                                 alpha=0.3, color=color)
        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            for b_index, b in enumerate(batchsize):
                for m_index, m in enumerate(iteration):
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            if b != 1 or m != 100:
                                continue
                        else:
                            if b != best_param_b or m != best_param_m:
                                continue
                    # print(name_algo, b, m)
                    time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                    n_samples_s_temp = n_samples_s.copy()
                    mask = np.ones(len(n_samples_s), dtype=bool)
                    # n_samples_s_temp = n_samples_s_temp.astype(float)
                    for i in range(len(n_samples_s)):
                        acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)
                        # print(params[acces_dict])
                        # print(acces_dict)
                        if params[acces_dict] is None:
                            # n_samples_s_temp[i] = np.NaN
                            mask[i] = 0
                            continue
                        time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                        time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                        time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                        time_list[i] = time_list_only_T[i] + time_list_only_W[i]
                    # print(mask)
                    # linestyle = "-"
                    plt.figure(0, figsize=figsize)
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            color = (1, 0.5, 0)
                            marker = "X"
                        else:
                            color = (1, 0, 0)
                            marker = "s"
                    else:
                        color = (1, b_index / len(batchsize), 0)

                    plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle,
                             label=name_algo_legend, color=color, marker=marker, markersize=markersize)
                    plt.fill_between(n_samples_s[mask],
                                     (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                                     (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                                     alpha=0.3, color=color)

                    plt.figure(1, figsize=figsize)

                    plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                             label=name_algo_legend, color=color, marker=marker, markersize=markersize)
                    plt.fill_between(n_samples_s[mask],
                                     (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                                     (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                                     alpha=0.3, color=color)
        else:
            # print(name_algo, name_algo_legend)
            time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            n_samples_s_temp = n_samples_s.copy()
            mask = np.ones(len(n_samples_s), dtype=bool)
            # n_samples_s_temp = n_samples_s_temp.astype(float)
            for i in range(len(n_samples_s)):
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
                # print(params[acces_dict])
                if params[acces_dict] is None:
                    # n_samples_s_temp[i] = np.NaN
                    mask[i] = 0
                    continue
                time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
                time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
                time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
                time_list[i] = time_list_only_T[i] + time_list_only_W[i]
            if "S_GWL" in name_algo:
                # linestyle = "--"
                color = (0, 0, 0)
                marker = "*"
            elif "uniform" in name_algo:
                # linestyle = "dotted"
                color = (0.1, 0.5, 0.1)
                marker = "d"
            elif "sliced_gromov" == name_algo:
                color = (0.40, 0.2, 0)
                # linestyle = "dotted"
                marker = "^"
            else:
                # linestyle = "--"
                color = (0, 0.8, 0.5)#(0, 0, 1)
                marker = "+"

            plt.figure(0, figsize=figsize)
            plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle, label=name_algo_legend,
                     color=color, marker=marker, markersize=markersize)
            plt.fill_between(n_samples_s[mask],
                             (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                             (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                             alpha=0.3,
                             color=color)

            plt.figure(1, figsize=figsize)
            if name_algo == "uniform" or name_algo == "identity":
                continue
            # print(name_algo)
            # for i in range(len(n_samples_s)):
            #     print(n_samples_s[i])
            #     print(np.mean(time_list_only_T, axis=1)[i], np.std(time_list_only_T, axis=1)[i])
            plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                     label=name_algo_legend, color=color, marker=marker, markersize=markersize)
            plt.fill_between(n_samples_s[mask],
                             (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                             (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                             alpha=0.3, color=color)

    plt.figure(0)
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=fontsize_axis)
    plt.ylabel('Computational time (s)', fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    if show_legend:
        plt.legend(loc=(1.01, 0.15), fontsize=fontsize_legend)
    # plt.title("Time needed to compute T and Wasserstein")

    plt.figure(1)
    plt.xlabel('Number of points: $N$', fontsize=fontsize_axis)
    plt.ylabel('Computational time (s)', fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    if show_legend:
        plt.legend(loc=(1.01, 0.15), fontsize=fontsize_legend)
    # plt.title("Time needed to compute T")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    n_samples_s_temp = [0] * len(names_algo)

    for a, name_algo in enumerate(names_algo):
        if name_algo in ["e_gromov", "e_gromov_KL"]:
            for e in entropy:
                W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
                W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                plt.figure(2, figsize=figsize)
                # print("start with algo", a, name_algo)
                # n_samples_s_temp[a] = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if params[acces_dict] is None:
                        mask[i] = False
                        continue
                    if n_samples_s[i] > 1000:
                        mask[i] = False
                    # if name_algo == "sliced_gromov":
                    #     W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 20
                    W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

                # linestyle = "--"
                if name_algo == "e_gromov":
                    color = (0, 0.5, 0.8)
                    marker = "o"
                if name_algo == "e_gromov_KL":
                    color = (0,0,1)#(0, 0.8, 0.5)
                    marker = "P"
                # print(np.mean(W_distance_list[mask], axis=1), name_algo, e)
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                         label=names_algo_legend[a] + legend_param, color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                 (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                 alpha=0.3,
                                 color=color)
        elif name_algo in ["S_GWL"]:
            for e in entropy_SGWL:
                W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
                W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
                plt.figure(2, figsize=figsize)
                # print("start with algo", a, name_algo)
                # n_samples_s_temp[a] = n_samples_s.copy()
                mask = np.ones(len(n_samples_s), dtype=bool)
                for i in range(len(n_samples_s)):
                    acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
                    if params[acces_dict] is None:
                        mask[i] = False
                        continue
                    if n_samples_s[i] > 1000:
                        mask[i] = False
                    # if name_algo == "sliced_gromov":
                    #     W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 20
                    W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

                # linestyle = "--"
                color = (0, 0, 0)
                marker = "*"
                # print(np.mean(W_distance_list[mask], axis=1), np.mean(np.mean(W_distance_list[mask], axis=1)), name_algo, e)
                if paper_plot:
                    legend_param = ""
                else:
                    legend_param = "$_{" + str(e) + "}$"
                plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                         label=names_algo_legend[a] + legend_param, color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                 (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                 alpha=0.3,
                                 color=color)
        elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced", "sampled_gromov_no_KL_e_constant"]:
            for b_index, b in enumerate(batchsize):
                for m_index, m in enumerate(iteration):
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            if b != 1 or m != 100:
                                continue
                        else:
                            if b != best_param_b or m != best_param_m:
                                continue
                    W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
                    W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))

                    plt.figure(2, figsize=figsize)

                    # print("start with algo", a, name_algo)
                    # n_samples_s_temp[a] = n_samples_s.copy()
                    mask = np.ones(len(n_samples_s), dtype=bool)
                    for i in range(len(n_samples_s)):
                        acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)
                        if params[acces_dict] is None:
                            # n_samples_s_temp[a][i] = np.NaN
                            # print("continue", name_algo, i)
                            mask[i] = False
                            continue
                        if n_samples_s[i] > 1000:
                            mask[i] = False
                        # if name_algo == "sliced_gromov":
                        #     W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 20
                        W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

                    # linestyle = "-"
                    if paper_plot:
                        if name_algo == "sampled_gromov_sliced":
                            color = (1, 0.5, 0)
                            marker = "X"
                        else:
                            color = (1, 0, 0)
                            marker = "s"
                    else:
                        color = (1, b_index / len(batchsize), 0)

                    plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                             label=names_algo_legend[a],
                             # label=None,
                             color=color, marker=marker, markersize=markersize)
                    plt.fill_between(n_samples_s[mask],
                                     (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                     (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                     alpha=0.3, color=color)
        else:
            W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
            W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
            plt.figure(2, figsize=figsize)
            # print("start with algo", a, name_algo)
            # n_samples_s_temp[a] = n_samples_s.copy()
            mask = np.ones(len(n_samples_s), dtype=bool)
            for i in range(len(n_samples_s)):
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
                if params[acces_dict] is None:
                    # n_samples_s_temp[a][i] = np.NaN
                    # print("continue", name_algo, i)
                    mask[i] = False
                    continue
                if n_samples_s[i] > 1000:
                    mask[i] = False
                if name_algo == "sliced_gromov":
                    W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"]
                W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

            if "S_GWL" in name_algo:
                # linestyle = "--"
                marker = ""
                color = (0, 0, 0)
            elif "uniform" in name_algo:
                # linestyle = "-."
                marker = "d"
                color = (0.9, 0.5, 0.9)
            elif "identity" in name_algo:
                # linestyle = "-."
                marker = ""
                color = (0.40, 0.2, 0)

            elif "sliced_gromov" in name_algo:
                # linestyle = "dotted"
                color = (0.40, 0.2, 0)
                marker = "^"

            else:
                marker = "+"
                # linestyle = "--"
                color = (0, 0.8, 0.5)#(0, 0, 1)

            if name_algo == "sliced_gromov":
                plt.plot(n_samples_s[mask], np.mean(W_distance_list_approx, axis=1)[mask], linestyle=linestyle,
                         label=names_algo_legend[a], color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list_approx, axis=1) - np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 (np.mean(W_distance_list_approx, axis=1) + np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 alpha=0.3, color=color)
                W_distance_list_approx = W_distance_list_approx * 25
                plt.plot(n_samples_s[mask], np.mean(W_distance_list_approx, axis=1)[mask], linestyle=linestyle,
                         label="25" + names_algo_legend[a], color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list_approx, axis=1) - np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 (np.mean(W_distance_list_approx, axis=1) + np.std(W_distance_list_approx, axis=1))[
                                     mask],
                                 alpha=0.3, color=color)
            else:
                plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                         label=names_algo_legend[a], color=color, marker=marker, markersize=markersize)
                plt.fill_between(n_samples_s[mask],
                                 (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                                 (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                                 alpha=0.3, color=color)

                # here just to add Sliced GW in the legend. Kind of ugly...

                if "gaussian_graph" == name_dataset and paper_plot and legend_SGW:
                    # linestyle = "dotted"
                    color = (0.40, 0.2, 0)
                    plt.plot([None], [None], linestyle=linestyle,
                             label="SGW", color=color)

                    plt.plot([None], [None], linestyle=linestyle,
                             label="15 SGW", color=color)
                    legend_SGW = False
    if log_scale:
        plt.xscale("log")
    plt.xlabel('Number of points: $N$', fontsize=fontsize_axis)
    plt.ylabel('GW value: $\mathcal{E}$ (T)', fontsize=fontsize_axis)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    # if name_dataset == "gaussian_point_graph":
    #     loc = (1, 0)
    # else:
    #     loc = (1, 0)
    # if show_legend:
    #     plt.legend(loc=(-0.8, 0.05), fontsize=fontsize_legend)
    if show_legend:
        if name_dataset == "gaussian_graph":
            plt.legend(loc=(1.01, 0.15), fontsize=fontsize_legend)
        else:
            plt.legend(loc=(1.01, 0.05), fontsize=fontsize_legend)

    # plt.title("Wasserstein distance")

    # if name_dataset == "same_graph":
    #     # n_samples_s_temp = [0] * len(names_algo)
    #     diff_to_identity_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10000
    #     plt.figure(3, figsize=figsize)
    #     for a, name_algo in enumerate(names_algo):
    #         if name_algo in ["e_gromov", "e_gromov_KL"]:
    #             for e_index, e in enumerate(entropy):
    #                 # n_samples_s_temp[a] = n_samples_s.copy()
    #                 mask = np.ones(len(n_samples_s), dtype=bool)
    #                 for i in range(len(n_samples_s)):
    #                     acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(e)
    #                     if params[acces_dict] is None:
    #                         # n_samples_s_temp[a][i] = np.NaN
    #                         mask[i] = False
    #                         continue
    #                     diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
    #                 linestyle = "--"
    #                 color = (0, e_index / (len(entropy)), 0)
    #                 if e_index == 0:
    #                     plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
    #                              label=names_algo_legend[a], color=color)
    #                     plt.fill_between(n_samples_s[mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      alpha=0.3, color=color)
    #                 else:
    #                     plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
    #                              color=color)
    #                     plt.fill_between(n_samples_s[mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                       axis=1))[
    #                                          mask],
    #                                      alpha=0.3, color=color)
    #
    #         elif name_algo in ["sampled_gromov_e_constant", "sampled_gromov_sliced"]:
    #             for b_index, b in enumerate(batchsize):
    #                 for m_index, m in enumerate(iteration):
    #                     if paper_plot:
    #                         if name_algo == "sampled_gromov_sliced":
    #                             if b != 1 or m != 100:
    #                                 continue
    #                         else:
    #                             if b != best_param_b or m != best_param_m:
    #                                 continue
    #                     # n_samples_s_temp[a] = n_samples_s.copy()
    #                     mask = np.ones(len(n_samples_s), dtype=bool)
    #                     for i in range(len(n_samples_s)):
    #                         acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i]) + str(b) + str(m)
    #                         if params[acces_dict] is None:
    #                             # n_samples_s_temp[a][i] = np.NaN
    #                             mask[i] = False
    #                             continue
    #                         diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
    #                     # if "sampled_gromov" in name_algo:
    #                     linestyle = "-"
    #                     # elif "S_GWL" in name_algo:
    #                     #     linestyle = "--"
    #                     # elif "sliced_gromov" in name_algo:
    #                     #     linestyle = "--"
    #                     # elif "uniform" in name_algo:
    #                     #     linestyle = "dotted"
    #                     # else:
    #                     #     linestyle = "--"
    #                     color = (m_index / len(iteration), 0, 0)
    #                     if b_index == 0 and m_index == 0:
    #                         plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask],
    #                                  linestyle=linestyle,
    #                                  label=names_algo_legend[a], color=color)
    #                         plt.fill_between(n_samples_s[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          alpha=0.3, color=color)
    #                     else:
    #                         plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask],
    #                                  linestyle=linestyle,
    #                                  color=color)
    #                         plt.fill_between(n_samples_s[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list,
    #                                                                                           axis=1))[mask],
    #                                          alpha=0.3, color=color)
    #
    #         else:
    #             # n_samples_s_temp[a] = n_samples_s.copy()
    #             mask = np.ones(len(n_samples_s), dtype=bool)
    #             for i in range(len(n_samples_s)):
    #                 acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
    #                 print(acces_dict)
    #                 if params[acces_dict] is None:
    #                     # n_samples_s_temp[a][i] = np.NaN
    #                     mask[i] = False
    #                     continue
    #                 diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
    #             if "sampled_gromov" in name_algo:
    #                 linestyle = "-"
    #             elif "S_GWL" in name_algo:
    #                 linestyle = "--"
    #             elif "sliced_gromov" in name_algo:
    #                 linestyle = "--"
    #             elif "uniform" in name_algo:
    #                 linestyle = "dotted"
    #             else:
    #                 linestyle = "--"
    #
    #             plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
    #                      label=names_algo_legend[a])
    #             plt.fill_between(n_samples_s[mask],
    #                              (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list, axis=1))[mask],
    #                              (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list, axis=1))[mask],
    #                              alpha=0.3)
    #
    #     if log_scale:
    #         plt.xscale("log")
    #     plt.legend()
    # plt.title("Wasserstein distance")

    if save:
        plt.figure(0)
        # plt.gca().set_axis_off()
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig("./figure/hyper/hyper_T_and_W_" + name_dataset + "_" + loss_func_name + ".pdf",
                    bbox_inches="tight")
        plt.figure(1)
        plt.savefig("./figure/hyper/hyper_T_" + name_dataset + "_" + loss_func_name + ".pdf",
                    bbox_inches="tight")
        plt.figure(2)
        plt.savefig("./figure/hyper/hyper_distance_" + name_dataset + "_" + loss_func_name + ".pdf",
                    bbox_inches="tight")
        # if name_dataset == "same_graph":
        #     plt.figure(3)
        #     plt.savefig("./figure/hyper/hyper_diff_to_identity_" + name_dataset + "_" + loss_func_name + ".pdf",
        #                 bbox_inches="tight")
    plt.show()


def analyse_data(name_dataset="gaussian",
                 pickle_path="pickle_compare",
                 names_algo=["sampled_gromov"],
                 names_algo_legend=["sampled_gromov"],
                 pickle_name="",
                 n_samples_s=[50, 100, 200],
                 n_samples_t=[],
                 noise_graph=0.1,
                 loss_func_name="1_loss",
                 best_T_not_needed=True,
                 log_scale=True,
                 figsize=(10, 5),
                 save=False):
    n_samples_s = np.array(n_samples_s)
    if name_dataset in ["gaussian_graph", "same_graph"]:
        try:
            names_algo.remove("sliced_gromov")
            names_algo_legend.remove("Sliced GW")
        except:
            pass

    if name_dataset in ["gaussian_graph", "gaussian_point_graph"]:
        try:
            names_algo.remove("identity")
            names_algo_legend.remove("Identity")
        except:
            pass

    if not n_samples_t:
        n_samples_t = n_samples_s

    params = {}

    same_param = None

    for name_algo in names_algo:
        for i in range(len(n_samples_s)):
            acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])

            if name_dataset == "same_graph":
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(n_samples_s[i]) + \
                              "_" + str(n_samples_t[i]) + "_" + loss_func_name + "__" + str(noise_graph) + "_" + \
                              pickle_name + ".pickle"
            else:
                path_pickle = "./" + pickle_path + "/" + name_dataset + "/" + name_algo + "/" + str(n_samples_s[i]) + \
                              "_" + str(n_samples_t[i]) + "_" + loss_func_name + pickle_name + ".pickle"

            if path.exists(path_pickle):
                with open(path_pickle, "rb") as pickle_in:

                    params[acces_dict] = pickle.load(pickle_in)
                    if same_param is None:
                        same_param = params[acces_dict]
                    elif params[acces_dict]["name_algo"] != "uniform":
                        # Use this to check if every run use the same parameter
                        for test in ["n_iter_algo", "rdm_seed", "same_space"]:
                            if same_param[test] != params[acces_dict][test]:
                                print(params[acces_dict])
                                pass
                                print("Warning", test, same_param[test], params[acces_dict][test])
                            # assert same_param[test] == params[acces_dict][test]
            else:
                # print("outer")
                params[acces_dict] = None

    assert same_param is not None  # Maybe an error in the pickle file name
    for name_algo, name_algo_legend in zip(names_algo, names_algo_legend):
        # print(name_algo, name_algo_legend)
        time_list_only_W = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        time_list_only_T = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        time_list_only_W_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        time_list = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        n_samples_s_temp = n_samples_s.copy()
        mask = np.ones(len(n_samples_s), dtype=bool)
        # n_samples_s_temp = n_samples_s_temp.astype(float)
        for i in range(len(n_samples_s)):
            acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
            # print(params[acces_dict])
            if params[acces_dict] is None:
                # n_samples_s_temp[i] = np.NaN
                mask[i] = 0
                continue
            time_list_only_W_approx[i] = params[acces_dict]["time_list_only_W_approx"]
            time_list_only_T[i] = params[acces_dict]["time_list_only_T"]
            time_list_only_W[i] = params[acces_dict]["time_list_only_W"]
            time_list[i] = time_list_only_T[i] + time_list_only_W[i]
        if "sampled_gromov" in name_algo:
            linestyle = "-"
        elif "S_GWL" in name_algo:
            linestyle = "--"
        elif "sliced_gromov" in name_algo:
            linestyle = "--"
        elif "uniform" in name_algo:
            linestyle = "dotted"
        else:
            linestyle = "--"
        plt.figure(0, figsize=figsize)
        plt.plot(n_samples_s[mask], np.mean(time_list, axis=1)[mask], linestyle=linestyle, label=name_algo_legend)
        plt.fill_between(n_samples_s[mask],
                         (np.mean(time_list, axis=1) - np.std(time_list, axis=1))[mask],
                         (np.mean(time_list, axis=1) + np.std(time_list, axis=1))[mask],
                         alpha=0.3)

        plt.figure(1, figsize=figsize)
        if name_algo == "uniform" or name_algo == "identity":
            continue
        # print(name_algo)
        # for i in range(len(n_samples_s)):
        #     print(n_samples_s[i])
        #     print(np.mean(time_list_only_T, axis=1)[i], np.std(time_list_only_T, axis=1)[i])
        plt.plot(n_samples_s[mask], np.mean(time_list_only_T, axis=1)[mask], linestyle=linestyle,
                 label=name_algo_legend)
        plt.fill_between(n_samples_s[mask],
                         (np.mean(time_list_only_T, axis=1) - np.std(time_list_only_T, axis=1))[mask],
                         (np.mean(time_list_only_T, axis=1) + np.std(time_list_only_T, axis=1))[mask],
                         alpha=0.3)

    plt.figure(0)
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")

    plt.legend()
    # plt.title("Time needed to compute T and Wasserstein")

    plt.figure(1)
    plt.legend()
    # plt.title("Time needed to compute T")

    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    n_samples_s_temp = [0] * len(names_algo)

    for a, name_algo in enumerate(names_algo):
        W_distance_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10e10
        W_distance_list_approx = np.zeros((len(n_samples_s), same_param["n_iter_algo"]))
        plt.figure(2, figsize=figsize)
        # print("start with algo", a, name_algo)
        # n_samples_s_temp[a] = n_samples_s.copy()
        mask = np.ones(len(n_samples_s), dtype=bool)
        for i in range(len(n_samples_s)):
            acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
            if params[acces_dict] is None:
                # n_samples_s_temp[a][i] = np.NaN
                # print("continue", name_algo, i)
                mask[i] = False
                continue
            if n_samples_s[i] > 1000:
                mask[i] = False
            if name_algo == "sliced_gromov":
                W_distance_list_approx[i, :] = params[acces_dict]["W_distance_list_approx"] * 25
            W_distance_list[i, :] = params[acces_dict]["W_distance_list"]

        if "sampled_gromov" in name_algo:
            linestyle = "-"
        elif "S_GWL" in name_algo:
            linestyle = "--"
        elif "sliced_gromov" in name_algo:
            linestyle = "--"
        elif "uniform" in name_algo:
            linestyle = "dotted"
        else:
            linestyle = "--"

        if name_algo == "sliced_gromov":
            plt.plot(n_samples_s[mask], np.mean(W_distance_list_approx, axis=1)[mask], linestyle=linestyle,
                     label=names_algo_legend[a])
            plt.fill_between(n_samples_s[mask],
                             (np.mean(W_distance_list_approx, axis=1) - np.std(W_distance_list_approx, axis=1))[mask],
                             (np.mean(W_distance_list_approx, axis=1) + np.std(W_distance_list_approx, axis=1))[mask],
                             alpha=0.3)
        else:
            plt.plot(n_samples_s[mask], np.mean(W_distance_list[mask], axis=1), linestyle=linestyle,
                     label=names_algo_legend[a])
            plt.fill_between(n_samples_s[mask],
                             (np.mean(W_distance_list, axis=1) - np.std(W_distance_list, axis=1))[mask],
                             (np.mean(W_distance_list, axis=1) + np.std(W_distance_list, axis=1))[mask],
                             alpha=0.3)
    if log_scale:
        plt.xscale("log")
    plt.legend()
    # plt.title("Wasserstein distance")

    if name_dataset == "same_graph":
        # n_samples_s_temp = [0] * len(names_algo)
        diff_to_identity_list = np.ones((len(n_samples_s), same_param["n_iter_algo"])) * 10000
        plt.figure(3, figsize=figsize)
        for a, name_algo in enumerate(names_algo):
            # n_samples_s_temp[a] = n_samples_s.copy()
            mask = np.ones(len(n_samples_s), dtype=bool)
            for i in range(len(n_samples_s)):
                acces_dict = name_algo + str(n_samples_s[i]) + "_" + str(n_samples_t[i])
                if params[acces_dict] is None:
                    # n_samples_s_temp[a][i] = np.NaN
                    mask[i] = False
                    continue
                diff_to_identity_list[i, :] = params[acces_dict]["diff_to_identity"]
            if "sampled_gromov" in name_algo:
                linestyle = "-"
            elif "S_GWL" in name_algo:
                linestyle = "--"
            elif "sliced_gromov" in name_algo:
                linestyle = "--"
            elif "uniform" in name_algo:
                linestyle = "dotted"
            else:
                linestyle = "--"

            plt.plot(n_samples_s[mask], np.mean(diff_to_identity_list, axis=1)[mask], linestyle=linestyle,
                     label=names_algo_legend[a])
            plt.fill_between(n_samples_s[mask],
                             (np.mean(diff_to_identity_list, axis=1) - np.std(diff_to_identity_list, axis=1))[mask],
                             (np.mean(diff_to_identity_list, axis=1) + np.std(diff_to_identity_list, axis=1))[mask],
                             alpha=0.3)

        if log_scale:
            plt.xscale("log")
        plt.legend()
        # plt.title("Wasserstein distance")

    if save:
        plt.figure(0)
        plt.savefig("./figure/T_and_W_" + name_dataset + "_" + loss_func_name + ".pdf")
        plt.figure(1)
        plt.savefig("./figure/T_" + name_dataset + "_" + loss_func_name + ".pdf")
        plt.figure(2)
        plt.savefig("./figure/distance_" + name_dataset + "_" + loss_func_name + ".pdf")
        if name_dataset == "same_graph":
            plt.figure(3)
            plt.savefig("./figure/diff_to_identity_" + name_dataset + "_" + loss_func_name + ".pdf")

    # print(W_distance_list)
    # best_w = np.min(W_distance_list, axis=(0, 2), keepdims=True)
    # # print(best_w)
    # # print(best_w.shape)
    # # print(W_distance_list)
    # W_distance_list = W_distance_list - best_w
    # # print(W_distance_list)
    # for a, name_algo in enumerate(names_algo):
    #     if "sampled_gromov" in name_algo:
    #         linestyle = "--"
    #     else:
    #         linestyle = "-"
    #     plt.figure(3, figsize=figsize)
    #
    #     plt.plot(n_samples_s_temp[a], np.mean(W_distance_list[a], axis=1),
    #              linestyle=linestyle, label=name_algo)
    #     plt.fill_between(n_samples_s_temp[a],
    #                      np.mean(W_distance_list[a], axis=1) - np.std(W_distance_list[a], axis=1),
    #                      np.mean(W_distance_list[a], axis=1) + np.std(W_distance_list[a], axis=1),
    #                      alpha=0.3)
    #     plt.title("Approx Wasserstein distance")
    #
    #     plt.legend()
    plt.show()


def main(args):
    print(args)

    np.random.seed(args.rdm_seed)

    time_only_dataset = time.time()
    if not args.enough_space:
        global Cs_save
        Cs_save = np.empty((args.n_samples_s, args.n_samples_s))
        Cs_save.fill(np.nan)
        global Ct_save
        Ct_save = np.empty((args.n_samples_t, args.n_samples_t))
        Ct_save.fill(np.nan)

    Cs, Xs, Ys, Ct, Xt, Yt = create_dataset(args=args,
                                            name_dataset=args.name_dataset,
                                            n_samples_s=args.n_samples_s,
                                            n_samples_t=args.n_samples_t,
                                            same_s_and_t=args.same_space,
                                            dimension_s=args.dimension_s,
                                            dimension_t=args.dimension_t)

    time_only_dataset = time.time() - time_only_dataset

    loss_func = define_loss_function(args.loss_func_name)

    W_distance_list_approx = np.zeros(args.n_iter_algo)
    W_distance_list = np.zeros(args.n_iter_algo)
    diff_to_identity = np.zeros(args.n_iter_algo)

    time_list_only_T = np.zeros(args.n_iter_algo)
    time_list_only_W_approx = np.zeros(args.n_iter_algo)
    time_list_only_W = np.zeros(args.n_iter_algo)

    for i in range(args.n_iter_algo):
        T, W_distance_list_approx[i], time_list_only_T[i], time_list_only_W_approx[i] = solve_G_OT(
            name_algo=args.name_algo,
            C1=Cs,
            C2=Ct,
            Xs=Xs,
            Xt=Xt,
            loss_func=loss_func,
            args=args)
        if not args.enough_space:
            Ct_save.fill(np.nan)
            Cs_save.fill(np.nan)
        if i == 0 and (time_list_only_T[i] + time_list_only_W_approx[i]) > ((24 + 4) * 60 * 60):
            print("The algorithm will take too long")
            print("W_distance_list", W_distance_list)
            print("W_distance_list_approx", W_distance_list_approx)
            print("Time dataset", time_only_dataset)
            print("Time only T", time_list_only_T)
            print("Time only W approximation", time_list_only_W_approx)
            print("Time only real W", time_list_only_W)
            raise Exception("Free some space for the cluster")

        if args.n_samples_s <= 1000 and args.name_algo != "identity":
            time_before_run = time.time()
            W_distance_list[i] = compute_distance(T=T, C1=Cs, C2=Ct, loss=loss_func)
            time_list_only_W[i] = time.time() - time_before_run
        else:
            W_distance_list[i] = None
            time_list_only_W[i] = None

        if args.name_algo == "identity":
            W_distance_list[i] = W_distance_list_approx[i]

        if args.name_dataset == "same_graph" and args.name_algo != "identity":
            diff_to_identity[i] = 100 * np.mean((1 / args.n_samples_s) - T[np.arange(args.n_samples_s),
                                                                           np.arange(args.n_samples_s)])

    # print("W_distance_list", W_distance_list)
    print("Mean Wasserstein distance", np.mean(W_distance_list))
    print("Mean Wasserstein distance approximated", np.mean(W_distance_list_approx))
    # print("Time dataset", time_only_dataset)
    # print("Time only T", time_list_only_T)
    print("Mean computational time for finding T", np.mean(time_list_only_T))
    print("Mean computational time for the approximated Wasserstein distance", np.mean(time_list_only_W_approx))
    print("Mean computational time for the real Wasserstein distance", np.mean(time_list_only_W))
    return W_distance_list_approx, W_distance_list, time_only_dataset, time_list_only_T, time_list_only_W_approx, \
           time_list_only_W, diff_to_identity


def main_compute_distance(args, from_graphs, to_graphs):
    print(args)

    np.random.seed(args.rdm_seed)
    X, y = create_dataset(args=args,
                          name_dataset=args.name_dataset)
    if os.path.exists(
            "pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name + args.param1
            + "_W.pickle"):
        W, _ = load_pickle_graph(args, n=len(X))
    else:
        with open("pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name + args.param1
                  + "_W.pickle", "wb") as f:
            pickle.dump(args, f)
        W = np.zeros((len(X), len(X)))

    # unique_graph = np.unique(from_graphs + to_graphs)
    from_graphs = np.array(from_graphs)

    to_graphs = np.array(to_graphs)

    loss_func = define_loss_function(args.loss_func_name)

    W_distance_matrix_approx = np.zeros((len(X), len(X)))
    # time_matrix = np.zeros((len(X), len(X)))
    for from_index in from_graphs:
        C_from = X[from_index]

        print(".", end="")
        for to_index in to_graphs:
            # triangular superior and do not compute if it as already been computed
            if from_index > to_index or W[from_index, to_index] > 0:
                continue
            time_init = time.time()

            C_to = X[to_index]
            if "sampled_gromov" in args.name_algo:

                if len(C_from) >= len(C_to):  # For stability of compute_distance_sampling_both

                    def C_from_(array_ij, array_kl=None):
                        a = C_from[array_ij[0], array_kl]
                        return a.reshape(len(array_ij[0]), -1, 1)

                    def C_to_(array_ij, array_kl=None):
                        a = C_to[array_ij[0], array_kl]
                        return a.reshape(len(array_ij[0]), 1, -1)

                    args.n_samples_s = len(C_from)
                    args.n_samples_t = len(C_to)
                    _, W_distance_matrix_approx[from_index, to_index], _, _ = solve_G_OT(name_algo=args.name_algo,
                                                                                         C1=C_from_,
                                                                                         C2=C_to_,
                                                                                         Xs=None,
                                                                                         Xt=None,
                                                                                         loss_func=loss_func,
                                                                                         args=args)
                else:
                    def C_to_(array_ij, array_kl=None):
                        a = C_to[array_ij[0], array_kl]
                        return a.reshape(len(array_ij[0]), -1, 1)

                    def C_from_(array_ij, array_kl=None):
                        a = C_from[array_ij[0], array_kl]
                        return a.reshape(len(array_ij[0]), 1, -1)

                    args.n_samples_s = len(C_to)
                    args.n_samples_t = len(C_from)
                    _, W_distance_matrix_approx[from_index, to_index], _, _ = solve_G_OT(name_algo=args.name_algo,
                                                                                         C1=C_to_,
                                                                                         C2=C_from_,
                                                                                         Xs=None,
                                                                                         Xt=None,
                                                                                         loss_func=loss_func,
                                                                                         args=args)
            else:
                args.n_samples_s = len(C_from)
                args.n_samples_t = len(C_to)
                _, W_distance_matrix_approx[from_index, to_index], _, _ = solve_G_OT(name_algo=args.name_algo,
                                                                                     C1=C_from,
                                                                                     C2=C_to,
                                                                                     Xs=None,
                                                                                     Xt=None,
                                                                                     loss_func=loss_func,
                                                                                     args=args)

            dict_to_save = {"W": W_distance_matrix_approx[from_index, to_index],
                            "time": time.time() - time_init,
                            "from": from_index,
                            "to": to_index}

            if args.save_pickle:
                if os.path.exists(
                        "pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name
                        + args.param1 + "_W.pickle"):
                    with open("pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name
                              + args.param1 + "_W.pickle", "ab") as f:
                        pickle.dump(dict_to_save, f)
                else:
                    with open("pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name
                              + args.param1 + "_W.pickle", "wb") as f:
                        pickle.dump(args, f)
                        pickle.dump(dict_to_save, f)

    print("All distances has been saved")


def load_pickle_graph(args, n):
    with open("pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name + args.param1
              + "_W.pickle", "rb") as f:
        W = np.zeros((n, n))
        W_number_computed = np.zeros((n, n))
        time_list = []
        args2 = pickle.load(f)
        while True:
            try:
                dict_pickle = pickle.load(f)
                W[dict_pickle["from"], dict_pickle["to"]] += dict_pickle["W"]
                W_number_computed[dict_pickle["from"], dict_pickle["to"]] += 1
                time_list.append(dict_pickle["time"])
            except:
                print("All data as been loaded")
                break
    W[W_number_computed > 0] = W[W_number_computed > 0] / W_number_computed[W_number_computed > 0]
    assert np.allclose(W, np.triu(W))
    W = W + W.T - np.diag(np.diag(W))
    if time_list == []:
        return W, np.array([])
    return W, np.mean(time_list)


def compute_accuracy(gamma, C_svm, y_train, y_test, W_train, W_test):
    if gamma == -1:
        kernel_train = W_train
        kernel_test = W_test
    else:
        kernel_train = np.exp(-gamma * W_train)
        kernel_test = np.exp(-gamma * W_test)
    svm = sklearn.svm.SVC(C=C_svm,
                          kernel="precomputed")
    svm.fit(kernel_train, y_train)
    accuracy = svm.score(kernel_test, y_test)
    return accuracy


def main_cross_val(args):
    print(args)
    nb_split = 10

    np.random.seed(args.rdm_seed)
    X, y = create_dataset(args=args,
                          name_dataset=args.name_dataset)
    y = np.array(y, dtype=int)

    W, time_W = load_pickle_graph(args, n=len(X))

    # Temporary
    if args.from_graphs != "" and args.to_graphs != "":
        from_graphs = args.from_graphs.split(",")
        to_graphs = args.to_graphs.split(",")
        a = np.concatenate((np.arange(int(from_graphs[0]), int(from_graphs[1])),
                            np.arange(int(to_graphs[0]), int(to_graphs[1]))))
        X = a
        y = y[a]
        W = W[np.ix_(a, a)]

    indices = np.arange(len(X))
    gamma_list = []
    C_svm_list = []
    time_list = []
    cross_acc_list = []
    W_distance_list = []
    test_acc_list = []
    for split in range(nb_split):
        index_train, index_test, y_train, y_test = train_test_split(indices,
                                                                    y,
                                                                    test_size=0.30,
                                                                    random_state=12345 + split)
        # print(y)
        # print(y_test)
        # print(y_train)
        print(len(y_train), np.sum(y_train))
        print(len(y_test), np.sum(y_test))
        # print(index_train, index_test)
        W_train = W[np.ix_(index_train, index_train)]
        W_test = W[np.ix_(index_test, index_train)]

        gammas = [2 ** i for i in range(-10, 10 + 1)]  # [-1] is linear
        C_svms = [10 ** i for i in range(-7, 7 + 1)]  # Optimal Transport for structured data with application on graphs
        kf = KFold(n_splits=10, random_state=42, shuffle=True)
        acc = np.zeros((len(gammas), len(C_svms), 10))
        fold = 0
        for index_val_train, index_val_test in kf.split(index_train, y_train):
            # print(index_val_train)
            # print(index_val_test)
            y_val_train = y_train[index_val_train]
            y_val_test = y_train[index_val_test]
            # print(y_val_train)
            # print(y_val_test)

            time_init = time.time()
            for gamma in range(len(gammas)):
                # print(gammas[gamma])
                for C_svm in range(len(C_svms)):
                    # print(C_svms[C_svm])
                    acc[gamma, C_svm, fold] = compute_accuracy(gammas[gamma], C_svms[C_svm],
                                                               y_train=y_train[index_val_train],
                                                               y_test=y_train[index_val_test],
                                                               W_train=W_train[
                                                                   np.ix_(index_val_train, index_val_train)],
                                                               W_test=W_train[np.ix_(index_val_test, index_val_train)])
                    # print(acc[gamma, C_svm, fold])
            print("One fold done", time.time() - time_init)
            fold += 1
        acc = np.mean(acc, axis=2)
        # print(acc)
        best_gamma, best_C_svm = np.unravel_index(acc.argmax(), acc.shape)
        # print(acc.argmax())
        # print(np.max(acc))
        acc_test = compute_accuracy(gammas[best_gamma], C_svms[best_C_svm],
                                    y_train=y_train,
                                    y_test=y_test,
                                    W_train=W_train,
                                    W_test=W_test)
        # print(gamma_list)
        # print(gammas)
        # print(best_gamma)
        # print(best_gamma)
        gamma_list.append(gammas[best_gamma])
        C_svm_list.append(C_svms[best_C_svm])
        time_list.append(time_W)
        cross_acc_list.append(acc[best_gamma, best_C_svm])
        W_distance_list.append(np.mean(W))
        test_acc_list.append(acc_test)
    dict_best_param = {"gamma": gamma_list,
                       "C_svm": C_svm_list,
                       "time": time_list,
                       "cross_acc": cross_acc_list,
                       "W_distance": W_distance_list,
                       "test_acc": test_acc_list}
    if args.save_pickle:
        with open("pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name
                  + "_best.pickle", "wb") as f:
            pickle.dump(dict_best_param, f)
    print("best param computed")
    print(dict_best_param)


def main_cross_val_knn(args):
    print(args)

    np.random.seed(args.rdm_seed)
    X, y = create_dataset(args=args,
                          name_dataset=args.name_dataset)
    y = np.array(y, dtype=int)

    W, time_W = load_pickle_graph(args, n=len(X))

    dict_best_param = {"W": np.mean(W),  # Should be use with caution.
                       "W_std": np.std(W),  # Same, does not have a lot of sense.
                       "time": time_W
                       }
    W[np.arange(0, len(W)), np.arange(0, len(W))] = np.inf

    min_W = np.argmin(W, axis=0)

    acc = (y[min_W] == y).mean()

    dict_best_param["acc"] = acc

    if args.save_pickle:
        with open("pickle_matching/" + args.name_algo + "_" + args.name_dataset + "_" + args.pickle_name + args.param1
                  + "_KNN" + "_best.pickle", "wb") as f:
            pickle.dump(dict_best_param, f)
    print("best param computed")
    print(dict_best_param)


def analyse_classif(name_dataset="reddit_binary"):
    if name_dataset == "reddit_binary":
        latex_tabular = {"Dataset": [], "Train": [], "Test": [], "Time": []}
    else:
        latex_tabular = {"Dataset": [], "Train": [], "Distance": [], "Time": []}

    round_value = 2
    for path in sorted(os.listdir("pickle_matching/")):
        if "_best" in path and name_dataset == "reddit_binary":
            print(path)
            with open("pickle_matching/" + path, "rb") as f:
                dict_best_param = pickle.load(f)

            latex_tabular["Dataset"].append((path.split(name_dataset)[0][:-1] +
                                             path.split(name_dataset)[1][1:-12]).replace("_", " "))
            latex_tabular["Train"].append("$" + str(np.round(np.mean(dict_best_param["cross_acc"]), round_value)) +
                                          " \\pm " + str(np.round(np.std(dict_best_param["cross_acc"]), round_value)) +
                                          "$")
            latex_tabular["Test"].append("$" + str(np.round(np.mean(dict_best_param["test_acc"]), round_value)) +
                                         " \\pm " + str(np.round(np.std(dict_best_param["test_acc"]), round_value)) +
                                         "$")
            latex_tabular["Time"].append("$" + str(np.round(np.mean(dict_best_param["time"]), round_value)) + "$")
        elif "_KNN_best" in path:  # and "exp" not in path:
            power = "p"
            with open("pickle_matching/" + path, "rb") as f:
                dict_best_param = pickle.load(f)

            param_algo = path.split(name_dataset)[1][1:-16]
            if "square_loss" in param_algo:
                param_algo = "2_"
            elif "square_root" in param_algo:
                param_algo = "05"
            if path.split(name_dataset)[0][:-1] == "sampled_gromov_e_constant":
                if "100" not in path:
                    continue
                param_algo_ = param_algo[:2]
                if param_algo_[1] == "_":
                    param_algo = param_algo_[0]
                else:
                    param_algo = param_algo_[0] + "." + param_algo_[1]
                if "exp" not in path:
                    param_algo = power + "=" + param_algo
                    # print(param_algo)
                    name_algo = ("\\SGW$_{" + param_algo + "}$")
                else:
                    name_algo = ("\\SGW " + path.split(name_dataset)[1][1:-13])
            elif path.split(name_dataset)[0][:-1] == "sampled_gromov_sliced":
                param_algo_ = param_algo[:2]
                if param_algo_[1] == "_":
                    param_algo = param_algo_[0]
                else:
                    param_algo = param_algo_[0] + "." + param_algo_[1]
                if "exp" not in path:
                    param_algo = power + "=" + param_algo
                    name_algo = ("\\PGW$_{" + param_algo + "}$")
                else:
                    name_algo = ("\\PGW " + path.split(name_dataset)[1][1:-13])

            elif path.split(name_dataset)[0][:-1] == "uniform":
                name_algo = "Uniform"
            elif path.split(name_dataset)[0][:-1] == "e_gromov":
                name_algo = "\\EGW{" + param_algo + "}"
            elif path.split(name_dataset)[0][:-1] == "e_gromov_KL":
                name_algo = "\\EGWKL{" + param_algo + "}"  # "\\EGWKL{}"
            elif path.split(name_dataset)[0][:-1] == "S_GWL":
                name_algo = "\\ScalableGW$_{" + param_algo + "}$"
            elif path.split(name_dataset)[0][:-1] == "gromov":
                name_algo = "\\EGW{0}"
            latex_tabular["Dataset"].append(name_algo)
            latex_tabular["Train"].append("$" + str(np.round(dict_best_param["acc"], round_value)) + "$")
            # " \\pm " + str(np.round(dict_best_param["acc_std"], round_value)) +
            # "$")
            if path.split(name_dataset)[0][:-1] in ["e_gromov", "e_gromov_KL", "uniform", "gromov",
                                                    "S_GWL"] or "square_loss" in path:
                # latex_tabular["Distance"].append("$" + str(np.round(np.mean(dict_best_param["W"]), 4)) +"$")
                latex_tabular["Distance"].append("$" + str(int(np.round(np.mean(dict_best_param["W"]), 0))) + "$")
                # " \\pm " + str(np.round(dict_best_param["W_std"], round_value)) +

            else:
                latex_tabular["Distance"].append("")
            latex_tabular["Time"].append("$" + str(np.ceil(np.mean(dict_best_param["time"]) * 10) / 10) + "$")

    df = pd.DataFrame(latex_tabular)
    print("\\begin{table*}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table*}")


def small_entropy_experiment(args):
    latex_tabular = {"epsilon": [], "SGW": [], "EGW": []}
    for e in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]:
        args.epsilon = e
        latex_tabular["epsilon"].append(str(e))
        for name_algo in ["sampled_gromov_e_constant", "e_gromov"]:
            args.name_algo = name_algo
            W_distance_list_approx, W_distance_list, time_only_dataset, time_list_only_T, time_list_only_W_approx, \
            time_list_only_W, diff_to_identity = main(args)
            if name_algo == "sampled_gromov_e_constant":
                latex_tabular["SGW"].append(np.round(np.mean(W_distance_list), 2))
            else:
                latex_tabular["EGW"].append(np.round(np.mean(W_distance_list), 2))

    df = pd.DataFrame(latex_tabular)
    print("\\begin{table*}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table*}")


def small_alpha_experiment(args):
    iter_epsilons = [1, 10, 100, 1000]  # [10, 100]  #
    latex_tabular = {"Number Iterations": iter_epsilons}
    # Quite ugly, but those values are modified during the process...
    ps_out_s = args.ps_out_s
    p_in_s = args.p_in_s
    ps_out_t = args.ps_out_t
    p_in_t = args.p_in_t
    for alpha in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 1]:
        latex_tabular[str(alpha)] = []
        for iter_epsilon in iter_epsilons:
            args.iter_epsilon = iter_epsilon
            args.learning_step = alpha
            args.ps_out_s = ps_out_s
            args.p_in_s = p_in_s
            args.ps_out_t = ps_out_t
            args.p_in_t = p_in_t
            # args.rdm_seed = int(1000 * alpha * iter_epsilon + 45)
            W_distance_list_approx, W_distance_list, time_only_dataset, time_list_only_T, time_list_only_W_approx, \
            time_list_only_W, diff_to_identity = main(args)
            latex_tabular[str(alpha)].append(np.round(np.mean(W_distance_list), 4) * 100)

    df = pd.DataFrame(latex_tabular)
    print("\\begin{table*}")
    print(df.to_latex(index=False, escape=False))
    print("\\end{table*}")
    raise SystemExit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GROMAP')
    parser.add_argument('--same_space', type=bool, default=False)
    parser.add_argument('--iter_max', type=int, default=1)
    parser.add_argument('--iter_epsilon', type=int, default=1000)
    parser.add_argument('--nb_iter_batch', type=int, default=1)
    parser.add_argument('--n_samples_s', type=int, default=100)
    parser.add_argument('--n_samples_t', type=int, default=0)
    parser.add_argument('--clique_size_s', type=int, default=200)
    parser.add_argument('--clique_size_t', type=int, default=0)
    parser.add_argument('--p_in_s', type=float, default=0.5)
    parser.add_argument('--p_in_t', type=float, default=0)
    parser.add_argument('--ps_out_s', type=float, default=0.1)
    parser.add_argument('--ps_out_t', type=float, default=0)
    parser.add_argument('--variance_s', type=float, default=5)
    parser.add_argument('--variance_t', type=float, default=0)
    parser.add_argument('--value_graph', type=bool, default=True)
    parser.add_argument('--sparse_graph', type=bool, default=True)
    parser.add_argument('--noise_graph', type=float, default=0.1)
    parser.add_argument('--loss_func_name', type=str, default="square_loss")

    parser.add_argument('--epsilon', type=float, default=1.0)
    parser.add_argument('--epsilon_min', type=float, default=0.)
    parser.add_argument('--KL', type=float, default=1)
    parser.add_argument('--learning_step', type=float, default=0.8)
    parser.add_argument("-r", '--repeat', default=True, action="store_false")
    parser.add_argument('--constraint', type=bool, default=True)
    parser.add_argument('--batch_size', default=None)
    parser.add_argument("-e", '--enough_space', default=True, action="store_false")

    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--nb_iter_update_Y', type=int, default=400)

    parser.add_argument('--name_dataset', type=str, default="gaussian")
    parser.add_argument('--dimension_s', type=int, default=20)
    parser.add_argument('--dimension_t', type=int, default=10)
    parser.add_argument('--name_algo', type=str, default="sampled_gromov")

    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--rdm_seed', type=int, default=12345)
    parser.add_argument('--pickle_path', type=str, default="pickle_compare")
    parser.add_argument('--pickle_name', type=str, default="")
    parser.add_argument('--param1', type=str, default="")
    parser.add_argument('--param2', type=str, default="")
    parser.add_argument('--barycenter', type=bool, default=False)
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--n_iter_algo', type=int, default=10)

    parser.add_argument('--from_graphs', type=str, default="")
    parser.add_argument('--to_graphs', type=str, default="")
    parser.add_argument("-entropy", "--entropy_experiment", action="store_true")
    parser.add_argument("-alpha", "--alpha_experiment", action="store_true")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Will print more information about the run")
    parser.add_argument("-s", "--save_pickle", action="store_true",
                        help="Will save the in a pickle.")
    parser.add_argument("-c", "--cross_val", action="store_true",
                        help="Will cross validate the hyperparameters for graph matching")

    args = parser.parse_args()
    if args.batch_size is not None:
        args.batch_size = [[int(args.batch_size), None], [1, None]]

    # at least 2 community
    args.clique_size_s = min(args.clique_size_s, int(args.n_samples_s / 2))
    args.clique_size_t = min(args.clique_size_t, int(args.n_samples_t / 2))

    if args.n_samples_t == 0:
        args.n_samples_t = args.n_samples_s
    if args.clique_size_t == 0:
        args.clique_size_t = args.clique_size_s
    if args.p_in_t == 0:
        args.p_in_t = args.p_in_s
    if args.ps_out_t == 0:
        args.ps_out_t = args.ps_out_s
    if args.variance_t == 0:
        args.variance_t = args.variance_s

    # entropy_experiment = False
    if args.entropy_experiment:
        small_entropy_experiment(args)
    if args.alpha_experiment:
        small_alpha_experiment(args)

    if args.name_dataset in ["reddit_binary", "reddit_multi", "DB"]:
        if args.cross_val:
            if args.name_dataset == "DB":
                main_cross_val_knn(args)
            else:
                main_cross_val(args)
        else:
            from_graphs = args.from_graphs.split(",")
            to_graphs = args.to_graphs.split(",")
            if len(from_graphs) == 2:
                from_graphs = list(range(int(from_graphs[0]), int(from_graphs[1])))
            else:
                for i in range(len(from_graphs)):
                    from_graphs[i] = int(from_graphs[i])
            if len(to_graphs) == 2:
                to_graphs = list(range(int(to_graphs[0]), int(to_graphs[1])))
            else:
                for i in range(len(to_graphs)):
                    to_graphs[i] = int(to_graphs[i])
            main_compute_distance(args,
                                  from_graphs=from_graphs,
                                  to_graphs=to_graphs)

    else:
        W_distance_list_approx, W_distance_list, time_only_dataset, time_list_only_T, time_list_only_W_approx, \
        time_list_only_W, diff_to_identity = main(args)

        param = vars(args)
        param["W_distance_list_approx"] = W_distance_list_approx
        param["W_distance_list"] = W_distance_list
        param["diff_to_identity"] = diff_to_identity
        param["time_only_dataset"] = time_only_dataset
        param["time_list_only_T"] = time_list_only_T
        param["time_list_only_W_approx"] = time_list_only_W_approx
        param["time_list_only_W"] = time_list_only_W

        if args.save_pickle:
            if args.name_dataset == "same_graph":
                with open("./" + args.pickle_path + "/" +
                          args.name_dataset + "/" +
                          args.name_algo + "/" +
                          str(args.n_samples_s) + "_" + str(args.n_samples_t) + "_" + args.loss_func_name + "_"
                                                                                                            "_" + str(
                    args.noise_graph) + "_" + args.param1 + "_" + args.param2 + "_" +
                          args.pickle_name + ".pickle", "wb") as pickle_out:
                    pickle.dump(param, pickle_out)
            else:
                with open("./" + args.pickle_path + "/" +
                          args.name_dataset + "/" +
                          args.name_algo + "/" +
                          str(args.n_samples_s) + "_" + str(args.n_samples_t) + "_" + args.loss_func_name + "_"
                          + args.param1 + "_" + args.param2 + "_" + args.pickle_name + ".pickle", "wb") as pickle_out:
                    pickle.dump(param, pickle_out)
