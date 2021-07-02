# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein transport method
"""

# Code taken from POT and modified for the paper Sampled Gromov Wasserstein.

import numpy as np

from ot.bregman import sinkhorn
# from ot.da import sinkhorn_l1l2_gl
from ot.da import sinkhorn_lpl1_mm
from ot.utils import dist, UndefinedParameter
from ot.optim import cg
from ot.lp import emd
from numpy.random import default_rng
# from scipy.stats import rv_discrete
import ot
import time

import warnings

warnings.filterwarnings("error")


def init_matrix(C1, C2, p, q, loss_fun='square_loss'):
    if loss_fun == 'square_loss':
        def f1(a):
            return (a ** 2)

        def f2(b):
            return (b ** 2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * np.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return np.log(b + 1e-15)

    constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
                     np.ones(len(q)).reshape(1, -1))
    constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
                     np.dot(q.reshape(1, -1), f2(C2).T))
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T):
    A = -np.dot(hC1, T).dot(hC2.T)
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T):
    tens = tensor_product(constC, hC1, hC2, T)

    return np.sum(tens * T)


def gwggrad(constC, hC1, hC2, T):
    return 2 * tensor_product(constC, hC1, hC2,
                              T)  # [12] Prop. 2 misses a 2 factor


def update_square_loss(p, lambdas, T, Cs):
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.divide(tmpsum, ppt)


def update_kl_loss(p, lambdas, T, Cs):
    tmpsum = sum([lambdas[s] * np.dot(T[s].T, Cs[s]).dot(T[s])
                  for s in range(len(T))])
    ppt = np.outer(p, p)

    return np.exp(np.divide(tmpsum, ppt))


def GW_init_T(N, K, T_is_sparse=False):
    if T_is_sparse:
        T = np.arange(K)
        T = np.tile(T, int(N / K) + 1)
        T = T[:N]
    else:
        T = np.ones((N, K)) / (N * K)
    if N == K:
        if T_is_sparse:
            T = np.arange(N)
        else:
            T = np.eye(N)
    return T


# def GW_update_T(C1, C2, loss_fun, T,
#                 iter_epsilon=1, nb_iter_batch=10,
#                 batch_size=None,
#                 constraint=False,
#                 epsilon_init=0,
#                 T_is_sparse=False,
#                 KL=False,
#                 nb_iter_global=(0, 1),
#                 verbose=False,
#                 epsilon_min=0.01):
#     stop = False
#
#     if batch_size is None:
#         batch_size = [C1.shape[0], C1.shape[1], C2.shape[0], C2.shape[1]]
#     else:
#         for i in range(len(batch_size)):
#             if batch_size[i] is None:
#                 batch_size[i] = np.inf
#
#     for i in range(2):
#         batch_size[i] = min(batch_size[i], C1.shape[i])
#         batch_size[i + 2] = min(batch_size[i + 2], C2.shape[i])
#     batch_size = np.array(batch_size, dtype=int)
#     rng = default_rng(np.random.randint(0, 10000))
#
#     for e in range(iter_epsilon):
#         epsilon = epsilon_init * ((1 - ((nb_iter_global[0] * iter_epsilon) + e)
#                                    / (nb_iter_global[1] * iter_epsilon)) ** 3 + epsilon_min)
#         print("epsilon", epsilon)
#         for iter_batch in range(nb_iter_batch):
#             previous_T = T.copy()
#
#             if T_is_sparse:
#                 i_pos = rng.choice(C1.shape[0], size=batch_size[0], replace=False)
#                 j_pos = rng.choice(C1.shape[1], size=batch_size[1], replace=False)
#
#                 a = C2[:, np.newaxis, :]
#                 b = C1[:, j_pos, np.newaxis]
#                 L_jl = np.sum(loss_fun(b[i_pos], a[T[i_pos]]), axis=0) / batch_size[0]
#                 loss = np.sum(L_jl[np.arange(batch_size[1]), T[j_pos]]) / batch_size[0]
#
#                 if verbose:
#                     print(iter_batch, ":", loss, "(batched)")
#
#                 if not constraint:
#                     if epsilon == 0:
#                         T[j_pos] = np.argmin(L_jl, axis=1)
#                     else:
#                         new_T = np.exp(-L_jl / epsilon)
#                         new_T /= np.sum(new_T, axis=1, keepdims=True)
#                         T[j_pos] = sparsification_of_T(new_T)
#                 else:
#                     if epsilon == 0:
#                         new_T = emd(a=np.ones(L_jl.shape[0]) / L_jl.shape[0],
#                                     b=np.ones(L_jl.shape[1]) / L_jl.shape[1],
#                                     M=L_jl)
#                     else:
#                         new_T = sinkhorn(a=np.ones(L_jl.shape[0]) / L_jl.shape[0],
#                                          b=np.ones(L_jl.shape[1]) / L_jl.shape[1],
#                                          M=L_jl,
#                                          reg=epsilon)
#
#                     T[j_pos] = sparsification_of_T(new_T)
#             else:
#                 if True:
#                     repeat = False
#                     if repeat:
#                         i_pos = np.random.randint(0, C1.shape[0], batch_size[0])
#                         if batch_size[2] != 1:
#                             print("The sample is not perfect as batch_size[2] is not egal to 1")
#                     else:
#                         i_pos = rng.choice(C1.shape[0], size=batch_size[0], replace=False)
#                     # Maybe not the smartest way... avoid a loop but the search for the position is in O(n) !
#                     T_cumsum = T[i_pos].cumsum(axis=1) * len(T)
#                     rdm_integer_i = np.random.rand(batch_size[0], 1, batch_size[2])
#                     k_pos = T_cumsum[:, :, np.newaxis] < rdm_integer_i
#                     # print(rdm_integer_i)
#                     k_pos = k_pos.sum(axis=1).reshape(-1)
#                     i_pos = np.repeat(i_pos, batch_size[2])
#                 else:
#                     index = [[0] * (dimension_OT - 1), [0] * (dimension_OT - 1)]
#                     for d in range(dimension_OT - 1):
#                         index[0][d] = np.random.choice(T.shape[0],
#                                                        size=batch_size[0, d],
#                                                        replace=repeat)
#                         for index_0_d_i in index[0][d]:
#                             index[1][d] = np.random.choice(T.shape[1],
#                                                            size=batch_size[1, d],
#                                                            replace=repeat,
#                                                            p=T[index_0_d_i, :])
#                 # i_pos = np.arange(C1.shape[0])
#                 # k_pos = np.arange(C2.shape[0])
#                 # k_pos = np.tile(k_pos, batch_size[2])
#                 #
#                 # i_pos = np.repeat(i_pos, batch_size[2])
#                 # T = np.eye(3)
#                 # print(i_pos)
#                 # print(k_pos)
#
#                 # For the moment j and l are keeped full.
#                 # j_pos = rng.choice(C1.shape[1], size=batch_size[1], replace=False)
#                 # l_pos = rng.choice(C2.shape[1], size=batch_size[3], replace=False)
#                 # print("T",T)
#                 # L_ijkl = loss_fun(C1[:, :, np.newaxis, np.newaxis], C2[np.newaxis, np.newaxis, :, :])
#                 # L_jl = np.einsum("ijkl,ik->jl", L_ijkl, T)
#                 # print("my sum", L_ijkl[0, :, 0, :] + L_ijkl[1, :, 1, :] + L_ijkl[2, :, 2, :])
#                 # print("L_ijkl", L_jl)
#                 # print(L_jl)
#                 # L_jl_copy = L_jl.copy()
#                 # if verbose:
#                 #     loss = np.sum(L_jl * T)
#                 #     print("Loss", loss)
#                 # print("C1", C1)
#                 # print("C2", C2)
#                 if False:
#                     learning_step = np.sum(T[i_pos, k_pos])
#                 else:
#                     learning_step = (batch_size[0] * batch_size[2]) / (C1.shape[0] * C2.shape[0])
#
#                 if batch_size[0] == 1 and batch_size[2] == 1 and epsilon == 0 \
#                         and (KL is None or KL is False) and constraint:
#                     print("Very special case, kind of sliced, do not work in every case")
#                     C1_sort = C1[i_pos[0]].argsort()
#                     C2_sort = C1[k_pos[0]].argsort()
#
#                     n, m = C2.shape[0], C1.shape[0]
#                     weights = np.zeros((2, m))
#
#                     ratio = n / m
#
#                     # fix array that depend on the value of n and m. We are only interested at the order and at the
#                     # number of point that should be send from one distribution to the other.
#                     # n = 2, m = 5 give [0, 0, 0, 1, 1] for the perm_global and [., ., 1, ., . ] for 2
#                     # All the masse of the the first 2 points are send to the points 0.
#                     # and only a fraction (computed in weights) of the 3rd points are send to 0 because perm_global_2
#                     # has a value.
#                     # In fact perm_global_2 have some value at every position.
#                     # But it is associated with 0 in weights[1].
#
#                     # I recomande to take example (n = 7, m=16 for instance) to understand this part.
#                     arange_1 = np.int_(np.arange(1, n + 1, ratio)) / n
#                     arange_2 = np.arange(1, m + 1) / m
#                     perm_global = np.int_(np.arange(0, n, ratio))
#                     perm_global_2 = np.minimum(perm_global + 1, n - 1)
#
#                     a1_a2 = arange_1 - arange_2
#                     # weights[0, i] associated to the transport of
#                     # the points i in X1 to the points X2 define in perm_global
#                     # weights[1, i] associated to the transport of
#                     #  the points i in X1 to the points X2 define in perm_global_2
#                     weights[0] = (1 / m) * (0 <= a1_a2) + (1 / m + a1_a2) * (0 > a1_a2)
#                     weights[1] = np.abs(a1_a2) * (0 > a1_a2)
#
#                     new_T = np.zeros_like(T)
#                     new_T[C1_sort, C2_sort[perm_global]] = weights[0]
#                     new_T[C1_sort, C2_sort[perm_global_2]] += weights[1]
#
#                     T = (T + learning_step * new_T) / (1 + learning_step)
#                     continue
#
#                 a = C1[i_pos]
#                 b = C2[k_pos]
#                 L_jl = np.mean(loss_fun(a[:, :, np.newaxis], b[:, np.newaxis, :]), axis=0)
#
#                 # if True: #tempo
#                 #     L_jl = np.sum(loss_fun(a[:, :, np.newaxis], b[:, np.newaxis, :]) * \
#                 #     T[i_pos, k_pos].reshape(-1, 1, 1),
#                 #                   axis=0)
#                 loss = np.sum(L_jl * T)
#                 if verbose:
#                     print(iter_batch, ":", loss, "(batched)")
#                 # print("L_jl.sum()", L_jl)
#                 # print("DIVISION OF L", L_jl/L_jl_copy)
#                 if KL is not None:
#                     L_jl = L_jl - epsilon * KL * np.log(T)
#
#                 if not constraint:
#                     if epsilon == 0:
#                         new_T = (np.min(L_jl, axis=1, keepdims=True) == L_jl) * 1
#                         new_T = new_T / (new_T.min(axis=1, keepdims=True) * T.shape[0])
#                     else:
#                         new_T = np.exp(-L_jl / epsilon)
#                         new_T /= np.sum(new_T, axis=1, keepdims=True) * T.shape[0]
#                 else:
#                     if epsilon == 0:
#                         new_T = emd(a=np.ones(L_jl.shape[0]) / L_jl.shape[0],
#                                     b=np.ones(L_jl.shape[1]) / L_jl.shape[1],
#                                     M=L_jl)
#                     else:
#                         new_T = sinkhorn(a=np.ones(L_jl.shape[0]) / L_jl.shape[0],
#                                          b=np.ones(L_jl.shape[1]) / L_jl.shape[1],
#                                          M=L_jl,
#                                          reg=epsilon)
#                 new_T = new_T / new_T.sum()
#                 if KL is None or KL is False or epsilon == 0:
#                     # print("sum new_T", new_T.sum())
#                     # print("sum T", T.sum())
#
#                     T = (T + learning_step * new_T) / (1 + learning_step)
#                     # print(T)
#                 else:
#                     T = new_T
#
#                 print("Difference between 2 iterations", ((T - previous_T) ** 2).sum())
#             if np.all(previous_T == T):
#                 stop = True
#                 print("stop")
#                 break
#         if stop:
#             break
#     return T


def sparsify_T(T, number_point_to_keep=None, threshold=None):
    if threshold is None:
        if number_point_to_keep is None:
            if T.shape[0] == T.shape[1]:
                number_point_to_keep = T.shape[0]
            else:
                number_point_to_keep = T.shape[0] + T.shape[1] - 1
        index = np.unravel_index(np.argsort(-T.ravel())[:number_point_to_keep], T.shape)
    else:
        if threshold == 0:
            threshold = 1 / (np.max(T.shape) * 2)
        index = np.where((T * (T > threshold)) != 0)
    return index[0], index[1], T[index]


def compute_distance_sparse(C1, C2, loss_fun, T, dim_T):
    if isinstance(C1, np.ndarray):
        s = np.sum(loss_fun(np.squeeze(C1[np.ix_(T[0], T[0])]),
                            np.squeeze(C2[np.ix_(T[1], T[1])]))
                   * T[2][:, np.newaxis] * T[2][np.newaxis, :])
    else:
        s = np.sum(loss_fun(np.squeeze(C1([np.arange(dim_T[0])]))[np.ix_(T[0], T[0])],
                            np.squeeze(C2([np.arange(dim_T[1])]))[np.ix_(T[1], T[1])])
                   * T[2][:, np.newaxis] * T[2][np.newaxis, :])

    return s / np.sum(T[2])


def compute_distance_sampling_both(C1, C2, loss_fun, T, number_sample=None, std=False, std_total=False):

    if T.shape[0] < T.shape[1]:
        raise Exception("T.shape[0] should be higher than T.shape[1].")
    if number_sample is None:
        if T.shape[0] == T.shape[1]:
            number_sample = 1  # T.shape[0]
        else:
            number_sample = 2  # 2 * max(T.shape[0], T.shape[1])
    if std:
        number_sample = max(2, number_sample)

    index_k = np.zeros((T.shape[0], T.shape[0], number_sample), dtype=int)
    index_l = np.zeros((T.shape[0], T.shape[0], number_sample), dtype=int)
    list_value_sample = np.zeros((T.shape[0], T.shape[0], number_sample))

    # TODO can we skip this loop ? ... not sure.
    for i in range(T.shape[0]):
        index_k[i] = np.random.choice(T.shape[1], size=(T.shape[0], number_sample), p=T[i, :] * T.shape[0])
        index_l[i] = np.random.choice(T.shape[1], size=(T.shape[0], number_sample), p=T[i, :] * T.shape[0])
    if isinstance(C1, np.ndarray):
        for n in range(number_sample):
            list_value_sample[:, :, n] = loss_fun(C1,
                                                  (C2[index_k[:, :, n].reshape(-1),
                                                      index_l[:, :, n].T.reshape(-1)]).reshape(T.shape[0], T.shape[0]))
    else:
        for n in range(number_sample):
            list_value_sample[:, :, n] = loss_fun(C1([np.arange(T.shape[0])]).reshape(T.shape[0], T.shape[0]),
                                                  C2([index_k[:, :, n].reshape(-1)],
                                                     index_l[:, :, n].T.reshape(-1)).reshape(T.shape[0], T.shape[0]))
    if std:
        std_value = np.sum(np.std(list_value_sample, axis=2) ** 2) ** 0.5
        print(std_value / (T.shape[0] * T.shape[0]))
        if std_total:
            return np.mean(list_value_sample), std_value / (T.shape[0] * T.shape[0]), np.std(list_value_sample)
        else:
            return np.mean(list_value_sample), std_value / (T.shape[0] * T.shape[0])
    else:
        return np.mean(list_value_sample)


def compute_distance(T, C1, C2, loss):
    # TODO to be optimised
    T = T / np.sum(T)
    s = 0
    if isinstance(C1, np.ndarray):
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                s = s + np.sum(loss(C1[i, np.newaxis, :, np.newaxis],
                                    C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
    else:
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                s = s + np.sum(loss(C1(np.array([[i]])),
                                    C2(np.array([[k]]))), axis=0) * T[i, k]

    return np.sum(s * T)


def compute_L(C1, C2, loss_fun, T):
    if np.max(T.shape) > 100:
        s = 0
        for i in range(T.shape[0]):
            for k in range(T.shape[1]):
                s = s + np.sum(loss_fun(C1[i, np.newaxis, :, np.newaxis],
                                        C2[k, np.newaxis, np.newaxis, :]), axis=0) * T[i, k]
        return s
    else:
        return np.sum(loss_fun(C1[:, :, np.newaxis, np.newaxis], C2[np.newaxis, np.newaxis, :, :])
                      * T[:, np.newaxis, :, np.newaxis], axis=(0, 2))


def greenkhorn_method(a, b, M, reg, numItermax=10000, stopThr=1e-9, verbose=False,
                      log=False):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype=np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype=np.float64) / M.shape[1]

    dim_a = a.shape[0]
    dim_b = b.shape[0]

    # Next 3 lines equivalent to K= np.exp(-M/reg), but faster to compute
    K = np.empty_like(M)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)

    u = np.full(dim_a, 1. / dim_a)
    v = np.full(dim_b, 1. / dim_b)
    G = u[:, np.newaxis] * K * v[np.newaxis, :]

    viol = G.sum(1) - a
    viol_2 = G.sum(0) - b
    stopThr_val = 1

    if log:
        log = dict()
        log['u'] = u
        log['v'] = v

    for i in range(numItermax):
        i_1 = np.argmax(np.abs(viol))
        i_2 = np.argmax(np.abs(viol_2))
        m_viol_1 = np.abs(viol[i_1])
        m_viol_2 = np.abs(viol_2[i_2])
        stopThr_val = np.maximum(m_viol_1, m_viol_2)

        if m_viol_1 > m_viol_2:
            old_u = u[i_1]
            u[i_1] = a[i_1] / (K[i_1, :].dot(v))
            G[i_1, :] = u[i_1] * K[i_1, :] * v

            viol[i_1] = u[i_1] * K[i_1, :].dot(v) - a[i_1]
            viol_2 += (K[i_1, :].T * (u[i_1] - old_u) * v)

        else:
            old_v = v[i_2]
            v[i_2] = b[i_2] / (K[:, i_2].T.dot(u))
            G[:, i_2] = u * K[:, i_2] * v[i_2]
            viol += (-old_v + v[i_2]) * K[:, i_2] * u
            viol_2[i_2] = v[i_2] * K[:, i_2].dot(u) - b[i_2]

        if stopThr_val <= stopThr:
            break
    else:
        warnings.warn("Warning: Algorithm did not converge")

    if log:
        log['u'] = u
        log['v'] = v

    if log:
        return G, log
    else:
        return G


def update_T_without_KL(old_T, new_T, learning_step):
    return (1 - learning_step) * old_T + learning_step * new_T


def sliced_n_m(n, m):
    assert n >= m  # n should be higher than m

    T = np.zeros((n, m))

    pos_n = np.array([0, 1])
    new_weight = np.array([1 / n, 0])
    # Maybe we can do it with numpy, but it will be run only once for each Gromov, this is not a big deal.
    # Maybe T at the end can be sparse, but it won't change the overall complexity.
    for i in range(n):

        T[i, pos_n[0]] += new_weight[0]

        if pos_n[1] < m:
            T[i, pos_n[1]] += new_weight[1]

        total_sum_used = (1 / n) * (i + 1)

        if new_weight[1] > 0:
            pos_n += 1
            total_mass_left_at_pos = (1 / m) * (pos_n[0] + 1) - total_sum_used
        else:
            total_mass_left_at_pos = (1 / m) * (pos_n[0] + 1) - total_sum_used

        if total_mass_left_at_pos == 0:
            pos_n += 1
            new_weight = np.array([1 / n, 0])
        elif total_mass_left_at_pos < 1 / n:
            # pos_n don't moove
            new_weight = np.array([total_mass_left_at_pos, 1 / n - total_mass_left_at_pos])
        else:
            # pos_n don't moove
            new_weight = np.array([1 / n, 0])

    return T


def Generalisation_OT(C1, C2, loss_fun, T,  # C1 and C2 can be function
                      dimension_OT=2,  # 1 for OT 2 for Gromov
                      iter_epsilon=500,  # Number of out iterations
                      nb_iter_batch=1,  # Similar to iter_epsilon if the entropy is not modified
                      batch_size=None,  # Number of sampled matrices at each iterations
                      constraint=True,  # True = OT, False = no marginal constrainte
                      epsilon_init=0,  # Entropy will get smaller and smaller if set > 0
                      T_is_sparse=False,
                      KL=1,
                      nb_iter_global=(0, 1), # Used to defined the decay of entropy
                      verbose=False,
                      epsilon_min=1,  # Minimum value of entropy
                      repeat=True,  # The sampling can be with or without remise
                      labels_s=None,  # Labels used for OTDA
                      labels_t=None,  # Labels used for OTDA
                      eta=1,  # Parameter of OTDA
                      W_distance_needed=False,
                      greenkhorn=False,  # Variant of GreenKhorn, doesn't work well as the marginal are not respected
                      sliced=True,  # PoGroW algorithm if only one sample is used without any KL.
                      learning_step=0.8  # Step of the FW algorithm, used by PoGroW
                      ):

    assert T.shape[0] >= T.shape[1]  # Just swap as preprocess and transpose the plan T after.
    if greenkhorn:
        assert labels_s is None  # greenkhorn is not implemented for OTDA.
    time_init = time.time()
    assert type(KL) is int
    continue_loop = True
    if batch_size is None:
        batch_size = [[1] * dimension_OT, [1] * dimension_OT]
        batch_size[0][-1], batch_size[1][-1] = T.shape[0], T.shape[1]
    else:
        for d in range(dimension_OT):
            if batch_size[0][d] is None:
                batch_size[0][d] = T.shape[0]
            if batch_size[1][d] is None:
                batch_size[1][d] = T.shape[1]
    time_print = False
    T_sliced = None

    batch_size = np.array(batch_size, dtype=int)

    if time_print:
        print("Before the loop", time.time() - time_init)
    for e in range(iter_epsilon):

        epsilon = epsilon_init * ((1 - ((nb_iter_global[0] * iter_epsilon) + e)
                                   / (nb_iter_global[1] * iter_epsilon)) ** 3) + epsilon_min
        for iter_batch in range(nb_iter_batch):
            if time_print:
                print("Iter(", e, iter_batch, "):")
                print(time.time() - time_init)

            index = [[0] * (dimension_OT - 1), [0] * (dimension_OT - 1)]
            for d in range(dimension_OT - 1):
                index[0][d] = np.random.choice(T.shape[0],
                                               size=batch_size[0, d],
                                               replace=repeat)
                index_temp = np.zeros(len(index[0][d]) * batch_size[1, d], dtype=int)
                for i, index_0_d_i in enumerate(index[0][d]):
                    index_temp[i * batch_size[1, d]:(i + 1) * batch_size[1, d]] = \
                        np.random.choice(T.shape[1],
                                         size=batch_size[1, d],
                                         p=T[index_0_d_i, :] * T.shape[0])
                index[1][d] = index_temp
                index[0][d] = np.repeat(index[0][d], batch_size[1, d])

            if np.prod(batch_size[:, :-1]) == 1 and epsilon == 0 and \
                    constraint and sliced:
                if e == 0 and iter_batch == 0:
                    print("Point Gromov")
                # Notice that we don't take into account the loss function
                # This might be false if this function is not convex
                C1_sort = C1(np.array(index[0])).ravel().argsort()
                C2_sort = C2(np.array(index[1])).ravel().argsort()
                new_T = np.zeros_like(T)
                if T.shape[0] == T.shape[1]:
                    new_T[C1_sort, C2_sort] = 1 / T.shape[0]
                else:
                    if T_sliced is None:
                        T_sliced = sliced_n_m(T.shape[0], T.shape[1])
                    new_T[C1_sort] = T_sliced
                    new_T[:, C2_sort] = new_T.copy()
                Lii_ = None
                T = update_T_without_KL(old_T=T, new_T=new_T, learning_step=learning_step)
                continue
            else:
                if time_print:
                    print("Before Lii_ ", time.time() - time_init)

                if len(index[0][0]) == 1:
                    Lii_ = np.mean(loss_fun(C1(np.array(index[0])), C2(np.array(index[1]))),
                                   axis=tuple(range(dimension_OT - 1)))
                else:
                    Lii_ = 0
                    for i in range(len(index[0][0])):
                        Lii_ += np.mean(loss_fun(C1(np.array(index[0][0][i])[np.newaxis, np.newaxis]),
                                                 C2(np.array(index[1][0][i])[np.newaxis, np.newaxis])),
                                       axis=tuple(range(dimension_OT - 1)))


                if time_print:
                    print("After Lii_  ", time.time() - time_init)
                max_Lii_ = np.max(Lii_)
                if max_Lii_ == 0:
                    continue
                Lii_ = Lii_ / max_Lii_

                if epsilon * KL > 0:
                    log_T = np.log(np.clip(T, np.exp(-200), 1))
                    log_T[log_T == -200] = -np.inf
                    Lii_ = Lii_ - epsilon * KL * log_T

                if time_print:
                    print("Before OT   ", time.time() - time_init)

                if not constraint:
                    if epsilon == 0:
                        new_T = (np.min(Lii_, axis=1, keepdims=True) == Lii_) / (T.shape[0])
                    else:
                        new_T = np.exp(-Lii_ / epsilon)
                        new_T /= np.sum(new_T, axis=1, keepdims=True) * T.shape[0]
                else:
                    if epsilon == 0:
                        new_T = emd(a=np.ones(T.shape[0]) / T.shape[0],
                                    b=np.ones(T.shape[1]) / T.shape[1],
                                    M=Lii_)
                    else:
                        try:
                            if labels_s is None:
                                if greenkhorn:
                                    new_T = greenkhorn_method(a=np.ones(T.shape[0]) / T.shape[0],
                                                              b=np.ones(T.shape[1]) / T.shape[1],
                                                              M=Lii_,
                                                              reg=epsilon)
                                    # Greenkhorn doesn't return values that are precise enought for the sampling
                                    # Sometime negative
                                    new_T = new_T * (new_T > 0)
                                    # Sometime it doesn't sum to 1
                                    new_T[:, -1] = new_T[:, -1] + 1 / T.shape[0] - new_T.sum(axis=1)
                                else:
                                    new_T = sinkhorn(a=np.ones(T.shape[0]) / T.shape[0],
                                                     b=np.ones(T.shape[1]) / T.shape[1],
                                                     M=Lii_,
                                                     reg=epsilon)
                            else:
                                if labels_t is not None:
                                    for i in range(len(labels_t[0])):
                                        Lii_[labels_s != labels_t[0][i], labels_t[1][i]] = np.inf
                                new_T = sinkhorn_lpl1_mm(a=np.ones(T.shape[0]) / T.shape[0],
                                                         b=np.ones(T.shape[1]) / T.shape[1],
                                                         M=Lii_,
                                                         reg=epsilon,
                                                         labels_a=labels_s,
                                                         eta=eta)
                        except (RuntimeWarning, UserWarning):
                            continue_loop = False
                            print("Warning catched: Return last stable T")
                            break
                    if time_print:
                        print("After OT    ", time.time() - time_init)

                new_T = new_T / new_T.sum()

            if KL * epsilon <= 0:
                new_T = update_T_without_KL(old_T=T, new_T=new_T, learning_step=learning_step)

            if verbose:
                if iter_batch == 0:
                    print(e, ": Epsilon : ", epsilon)
                if iter_batch == (nb_iter_batch - 1):
                    print("||T - T|| : ", ((new_T - T) ** 2).sum(), )
                    if Lii_ is not None and Lii_.sum() != np.inf:
                        loss = np.sum(Lii_ * T)
                        print("Loss :", loss, "(batched)")
                        print("")
                    else:
                        print("")
            # FOR THE EXPERIMENT THAT ANALYSE THE NUMBER OF ITERATION.
            # print(compute_distance_sampling_both(C1=C1,
            #                                         C2=C2,
            #                                         loss_fun=loss_fun,
            #                                         T=T))
            # print("After computing distance :", time.time() - time_init)

            if ((T - new_T) ** 2).sum() <= 10e-20:  # 10
                continue_loop += 1
                if continue_loop > 100:  # Number max of low modification of T 10
                    continue_loop = False
                    print("Too many small modification of T")
                    T = new_T.copy()
                    break
            else:
                continue_loop = True
            T = new_T.copy()
        if not continue_loop:
            break
    if W_distance_needed:
        W_distance = compute_distance_sampling_both(C1=C1,
                                                    C2=C2,
                                                    loss_fun=loss_fun,
                                                    T=T)
        return T, W_distance
    return T


def entropic_gromov_wasserstein(C1, C2, p, q, loss_fun, epsilon,
                                max_iter=1000, tol=1e-9, verbose=False, log=False,
                                KL=False):
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    T = np.outer(p, q)  # Initialization
    if loss_fun in ["square_loss", "kl_loss"]:
        constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

    cpt = 0
    err = 1

    if log:
        log = {'err': []}
    time_init = time.time()

    while (err > tol and cpt < max_iter):
        # print(".", end="")

        Tprev = T

        # compute the gradient
        if loss_fun in ["square_loss", "kl_loss"]:
            tens = gwggrad(constC, hC1, hC2, T)
        else:
            tens = 2 * compute_L(C1=C1, C2=C2, loss_fun=loss_fun, T=T)

        if epsilon * KL > 0:
            log_T = np.log(np.clip(T, np.exp(-200), 1))
            log_T[log_T == -200] = -np.inf
            tens = tens - epsilon * KL * log_T

        if epsilon > 0:
            m = np.max(tens)
            try:
                T = sinkhorn(p, q, tens / m, epsilon)
            except:
                print("The method as not converged. Return last stable T. Nb iter : " + str(cpt))
                break
        else:
            try:
                T = emd(p, q, tens)
            except:
                print("The method as not converged. Return last stable T. Nb iter : " + str(cpt))
                break

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = np.linalg.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                        'It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(cpt, err))
        # print(cpt, time.time() - time_init)
        cpt += 1

    if log:
        if loss_fun in ["square_loss", "kl_loss"]:
            log['gw_dist'] = gwloss(constC, hC1, hC2, T)
        else:
            log['gw_dist'] = np.sum(T * compute_L(C1, C2, loss_fun, T))
        return T, log
    else:
        return T


def GW_update_T_full(C1, C2, loss_fun, T, epsilon=0, nb_iter_gromov=10, verbose=False,
                     constraint=False, KL=False):

    for i in range(nb_iter_gromov):
        L_ijkl = loss_fun(C1[:, :, np.newaxis, np.newaxis], C2[np.newaxis, np.newaxis, :, :])
        L_jl = np.einsum("ijkl,ik->jl", L_ijkl, T)
        # print(L_jl.sum())
        if verbose:
            loss = np.sum(L_jl * T)
            print("Loss", i, loss)
        if KL:
            if epsilon == 0:
                pass
            else:
                L_jl = L_jl - epsilon * np.log(T)

        if not constraint:
            if epsilon == 0:
                T = (np.min(L_jl, axis=1, keepdims=True) == L_jl) * 1
                T = T / np.sum(T)
            else:
                T = np.exp(-L_jl / epsilon)
                T /= np.sum(T, axis=1, keepdims=True)
                T = T / np.sum(T)
        else:
            if epsilon == 0:
                T = emd(a=np.ones(L_jl.shape[0]) / L_jl.shape[0],
                        b=np.ones(L_jl.shape[1]) / L_jl.shape[1],
                        M=L_jl)
            else:
                T = sinkhorn(a=np.ones(L_jl.shape[0]) / L_jl.shape[0],
                             b=np.ones(L_jl.shape[1]) / L_jl.shape[1],
                             M=L_jl,
                             reg=epsilon)
    return T


def GW_update_new_T(C1, C2, loss_fun, T, init_epsilon=0, nb_iter_gromov=10, verbose=False,
                    constraint=False, KL=False):
    epsilon = init_epsilon
    while True:
        print("epsilon:", epsilon)
        T = GW_update_T_full(C1, C2, loss_fun, T,
                             epsilon, nb_iter_gromov, verbose,
                             constraint, KL)
        epsilon = epsilon / 2
        if epsilon < 0.0001:
            break
    return T
