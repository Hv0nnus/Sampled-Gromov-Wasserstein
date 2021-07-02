import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import GROMOV_personal as gromov
import pickle
import time
import ot
import networkx as nx
import argparse


def main(entropy=True,
         epsilons=[0, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
         points=[20, 50, 100, 500]):

    if entropy:
        nb_iter = 1
        np.random.seed(12345)
        n_samples_s = 50
        n_samples_t = n_samples_s
        clique_size_s = 20
        clique_size_t = clique_size_s
        variance_s = 5
        variance_t = variance_s
        p_in_s = 0.5
        p_in_t = p_in_s
        ps_out_s = 0.1
        ps_out_t = ps_out_s

        # generate synthetic graph
        Gs = nx.gaussian_random_partition_graph(n=n_samples_s, s=clique_size_s,
                                                v=variance_s,
                                                p_in=p_in_t, p_out=ps_out_s,
                                                directed=False,
                                                seed=np.random.randint(10000))

        Gt = nx.gaussian_random_partition_graph(n=n_samples_t, s=clique_size_t,
                                                v=variance_t,
                                                p_in=p_in_t, p_out=ps_out_t,
                                                directed=False,
                                                seed=np.random.randint(10000))

        Cs, Ct = nx.adjacency_matrix(Gs).toarray(), nx.adjacency_matrix(Gt).toarray()
        Cs = Cs * np.random.uniform(0.25, 1.75, Cs.shape)
        Ct = Ct * np.random.uniform(0.25, 1.75, Ct.shape)
        Cs += Cs.T
        Ct += Ct.T

        def loss_fun(C1, C2):
            return np.abs(C1 - C2)

        distance = {}
        # epsilon_list = [0.1, 0.5, 1]
        for epsilon in epsilons:
            print("\nepsilon :", epsilon)

            T = gromov.entropic_gromov_wasserstein(C1=Cs,
                                                   C2=Ct,
                                                   p=ot.unif(n_samples_s),
                                                   q=ot.unif(n_samples_t),
                                                   loss_fun=loss_fun,
                                                   epsilon=epsilon)

            std = None
            t = []
            for i in range(nb_iter):
                np.random.seed(123 + i)
                time_init = time.time()
                d = gromov.compute_distance(T=T, C1=Cs, C2=Ct, loss=loss_fun)
                t.append(time.time() - time_init)
            distance[str(epsilon) + "real" + "distance"] = d
            distance[str(epsilon) + "real" + "std"] = std
            distance[str(epsilon) + "real" + "time"] = t

            std = None
            t = []
            for i in range(nb_iter):
                np.random.seed(123 + i)
                time_init = time.time()
                d, std, std_total = gromov.compute_distance_sampling_both(T=T, C1=Cs, C2=Ct,
                                                                          loss_fun=loss_fun,
                                                                          std=True,
                                                                          std_total=True)
                t.append(time.time() - time_init)
            distance[str(epsilon) + "sample" + "distance"] = d
            distance[str(epsilon) + "sample" + "std"] = std
            distance[str(epsilon) + "sample" + "std_total"] = std_total
            distance[str(epsilon) + "sample" + "time"] = t

            std = None
            t = []
            for i in range(nb_iter):
                np.random.seed(123 + i)
                time_init = time.time()
                T_ = gromov.sparsify_T(T)
                d = gromov.compute_distance_sparse(Cs, Ct, loss_fun,
                                                   T_, dim_T=T.shape)
                t.append(time.time() - time_init)
            distance[str(epsilon) + "sparse" + "distance"] = d
            distance[str(epsilon) + "sparse" + "std"] = std
            distance[str(epsilon) + "sparse" + "time"] = t
            print(distance)
        with open("pickle_distance/entropy_distance.pickle", 'wb') as handle:
            pickle.dump(distance, handle)
    else:
        nb_iter = 10

        def loss_fun(C1, C2):
            return (C1 - C2) ** 2

        distance = {}
        epsilon = 0.05
        for p, point in enumerate(points):
            print("\npoint", point)
            np.random.seed(123456 + p)

            n_samples_s = point
            n_samples_t = n_samples_s
            clique_size_s = int(point / 4)
            clique_size_t = clique_size_s
            variance_s = 5
            variance_t = variance_s
            p_in_s = 0.5
            p_in_t = p_in_s
            ps_out_s = 0.1
            ps_out_t = ps_out_s

            # generate synthetic graph
            Gs = nx.gaussian_random_partition_graph(n=n_samples_s, s=clique_size_s,
                                                    v=variance_s,
                                                    p_in=p_in_t, p_out=ps_out_s,
                                                    directed=False,
                                                    seed=np.random.randint(10000))

            Gt = nx.gaussian_random_partition_graph(n=n_samples_t, s=clique_size_t,
                                                    v=variance_t,
                                                    p_in=p_in_t, p_out=ps_out_t,
                                                    directed=False,
                                                    seed=np.random.randint(10000))

            Cs, Ct = nx.adjacency_matrix(Gs).toarray(), nx.adjacency_matrix(Gt).toarray()
            Cs = Cs * np.random.uniform(0.25, 1.75, Cs.shape)
            Ct = Ct * np.random.uniform(0.25, 1.75, Ct.shape)
            Cs += Cs.T
            Ct += Ct.T

            def Cs_(array_ij, array_kl=None):
                a = Cs[array_ij[0], array_kl]
                return a.reshape(len(array_ij[0]), -1, 1)

            def Ct_(array_ij, array_kl=None):
                a = Ct[array_ij[0], array_kl]
                return a.reshape(len(array_ij[0]), 1, -1)

            T = gromov.Generalisation_OT(C1=Cs_,
                                         C2=Ct_,
                                         loss_fun=loss_fun,
                                         T=np.outer(ot.unif(n_samples_s), ot.unif(n_samples_t)),
                                         epsilon_min=epsilon,  # swap between the epsilon !
                                         nb_iter_batch=1000,
                                         epsilon_init=0,
                                         KL=1)

            std = None
            t = []
            if point <= 500:
                for i in range(nb_iter):
                    np.random.seed(12345 * p + i)
                    time_init = time.time()
                    d = gromov.compute_distance(T=T, C1=Cs, C2=Ct, loss=loss_fun)
                    t.append(time.time() - time_init)
            else:
                t = d = None
            distance[str(point) + "real" + "distance"] = d
            distance[str(point) + "real" + "std"] = std
            distance[str(point) + "real" + "time"] = t

            std = None
            t = []
            for i in range(nb_iter):
                np.random.seed(12345 * p + i)
                time_init = time.time()
                d, std, std_total = gromov.compute_distance_sampling_both(T=T, C1=Cs, C2=Ct,
                                                                          loss_fun=loss_fun,
                                                                          std=True,
                                                                          std_total=True)
                t.append(time.time() - time_init)
            distance[str(point) + "sample" + "distance"] = d
            distance[str(point) + "sample" + "std"] = std
            distance[str(point) + "sample" + "std_total"] = std_total
            distance[str(point) + "sample" + "time"] = t

            std = None
            t = []
            for i in range(nb_iter):
                np.random.seed(12345 * p + i)
                time_init = time.time()
                T_ = gromov.sparsify_T(T)
                d = gromov.compute_distance_sparse(Cs, Ct, loss_fun, T_, dim_T=T.shape)
                t.append(time.time() - time_init)
            distance[str(point) + "sparse" + "distance"] = d
            distance[str(point) + "sparse" + "std"] = std
            distance[str(point) + "sparse" + "time"] = t

        with open("pickle_distance/points_distance.pickle", 'wb') as handle:
            pickle.dump(distance, handle)


def plot(log_scale=True, entropy=True, epsilon_range=None, figsize1=10, figsize2=10, marker=["o", "x", "^"]):
    if entropy:
        with open("pickle_distance/entropy_distance.pickle", 'rb') as handle:
            distance = pickle.load(handle)
        if epsilon_range is None:
            epsilon_range = [0.0, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # , 5, 10]
    else:
        with open("pickle_distance/points_distance.pickle", 'rb') as handle:
            distance = pickle.load(handle)
        if epsilon_range is None:
            epsilon_range = [10, 20, 50, 100, 500]
    name_algo2 = ["Real", "Sampled", "Sparse"]
    for n_i, name_algo in enumerate(["real", "sample", "sparse"]):
        time_list = []
        distance_list = []
        std_list = []
        std_total_list = []
        for epsilon in epsilon_range:
            if distance[str(epsilon) + name_algo + "time"] is not None:
                time_list.append(distance[str(epsilon) + name_algo + "time"])
                distance_list.append(distance[str(epsilon) + name_algo + "distance"])
                if name_algo == "sample":
                    std_total_list.append(distance[str(epsilon) + name_algo + "std_total"])
                std_list.append(distance[str(epsilon) + name_algo + "std"])

        time_list = np.array(time_list)
        distance_list = np.array(distance_list)
        std_list = np.array(std_list)
        std_total_list = np.array(std_total_list)
        plt.figure(1, figsize=(figsize1, figsize2))

        plt.plot(range(time_list.shape[0]), np.mean(time_list, axis=1), marker[n_i], label=name_algo2[n_i],
                                     linestyle="-", markersize=8)
        # plt.plot(range(time_list.shape[0]), np.mean(time_list, axis=1), marker[n_i], label=name_algo2[n_i])
        plt.fill_between(range(time_list.shape[0]),
                         np.mean(time_list, axis=1) - np.std(time_list, axis=1),
                         np.mean(time_list, axis=1) + np.std(time_list, axis=1),
                         alpha=0.3)
        # name_algo2[1] = "Sampled mean"
        plt.figure(2, figsize=(figsize1, figsize2))

        plt.plot(range(time_list.shape[0]), distance_list, marker[n_i], label=name_algo2[n_i], linestyle="-",
                 markersize=8)
        if std_list[0] is not None and n_i == 1:
            plt.fill_between(range(time_list.shape[0]),
                             distance_list - 2 * std_list,
                             distance_list + 2 * std_list,
                             alpha=0.3, color="orange")#, label="2 Standard deviation")
        # alpha = 0.3, label = "Std with stratification")
        if name_algo == "sample" and False:
            plt.fill_between(range(time_list.shape[0]),
                             distance_list - 0.25 * std_total_list,
                             distance_list + 0.25 * std_total_list,
                             alpha=0.1, color="black", label="Std without stratification")

    plt.figure(1)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 2]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 12})
    # plt.title("Time needed to compute T and Wasserstein")
    if entropy:
        plt.xlabel("Entropy: $\epsilon$", fontsize=14)
    else:
        plt.xlabel("Number of points: $N$", fontsize=14)
    plt.ylabel("Computational time (s)", fontsize=14)
    plt.xticks(range(len(epsilon_range)), epsilon_range, fontsize=10)
    plt.yticks(fontsize=10)
    if not entropy:
        plt.yscale("log")
    if entropy:
        plt.savefig("./figure/epsilon_time.pdf", bbox_inches="tight")
        # plt.savefig("./figure/epsilon_time.png", bbox_inches="tight")
    else:
        plt.savefig("./figure/point_time.pdf", bbox_inches="tight")
        # plt.savefig("./figure/point_time.png", bbox_inches="tight")


    plt.figure(2)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 2]
    # plt.title("Wasserstein Distance")
    if entropy:
        plt.xlabel("Entropy: $\epsilon$", fontsize=14)
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], prop={'size': 12})
    else:
        plt.xlabel("Number of points: $N$", fontsize=14)
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                   loc=2, prop={'size': 12})
    plt.ylabel("GW value : $\mathcal{E}$ (T)", fontsize=14)
    plt.xticks(range(len(epsilon_range)), epsilon_range, fontsize=10)
    plt.yticks(fontsize=10)
    if entropy:
        plt.savefig("./figure/epsilon_distance.pdf", bbox_inches="tight")
        # plt.savefig("./figure/epsilon_distance.eps", bbox_inches="tight")
    else:
        plt.savefig("./figure/point_distance.pdf", bbox_inches="tight")
        # plt.savefig("./figure/point_distance.eps", bbox_inches="tight")
    plt.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GROMAP')
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the result")
    parser.add_argument("-e", "--entropy", action="store_true", help="Use entropy experiment")
    parser.add_argument("--epsilons", default="0.0,0.005,0.01,0.05,0.1,0.5,1.0", help="epsilons values")
    parser.add_argument("--points", default="10,20,50,100,500", help="number of points")
    args = parser.parse_args()
    if args.plot:
        if args.entropy:
            list_range = args.epsilons
        else:
            list_range = args.points
        plot(entropy=args.entropy, epsilon_range=list_range)
    else:
        epsilons = args.epsilons.split(",")
        points = args.points.split(",")
        for i in range(len(epsilons)):
            epsilons[i] = float(epsilons[i])
        for i in range(len(points)):
            points[i] = int(points[i])
        main(entropy=args.entropy,
             epsilons=epsilons,
             points=points)
