import numpy as np
import sys
from timeit import default_timer as timer
from utils import read_g2o_file, split_measurements
from frankwolfe import grad_aug_alg_conn, fw_subset
import networkx as nx
import networkx.linalg.algebraicconnectivity as ac

def nx_rot_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.kappa
        G.add_edge(i, j, weight=w)
    return G

def nx_tran_G_w(measurements, num_poses):
    G = nx.generators.classic.empty_graph(num_poses)
    for measurement in measurements:
        i = measurement.i
        j = measurement.j
        w = measurement.tau
        G.add_edge(i, j, weight=w)
    return G

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [.g2o file]")
        sys.exit()

    print("Reading g2o file")
    start = timer()
    measurements, num_poses = read_g2o_file(sys.argv[1])
    end = timer()
    print("Success! elapsed time: ", (end - start))

    print("Constructing nx graph")
    start = timer()
    rot_G_w = nx_rot_G_w(measurements, num_poses)
    end = timer()
    print("Success! elapsed time: ", (end - start))
    print("Number of nodes: ", len(rot_G_w.nodes()))
    print("Number of edges: ", len(rot_G_w.edges()))

    print("Computing algebraic connectivity")
    start = timer()
    algconn = ac.algebraic_connectivity(rot_G_w, method='lanczos')
    end = timer()
    print("Success! elapsed time: ", (end - start))
    print("Alg. Conn: ", algconn)

    # Split measurements into odom and loop closures
    odom_measurements, lc_measurements = split_measurements(measurements)
    G_odom = nx_rot_G_w(odom_measurements, num_poses)
    G_lc = nx_rot_G_w(lc_measurements, num_poses)

    # Let's say we want 50% of the loop closures for now
    K = int(0.5 * len(lc_measurements))
    print(K)
    w_init = np.zeros(len(lc_measurements))
    # w_init[0:K] = 1.0
    # print(sum(w_init))
    grad_f = lambda w: grad_aug_alg_conn(G_odom, G_lc, w)
    w_final = fw_subset(None, grad_f, w_init, K, max_iters=100, debug_plot=False)
    print(w_final)
    print(sum(w_final))


    G_combined = nx.Graph()
    for u,v,a in G_odom.edges(data=True):
        G_combined.add_edge(u, v, weight=a['weight'])
    i = 0
    for u,v,a in G_lc.edges(data=True):
        # G_combined.add_edge(u,v,weight=w[i]*a['weight'])
        # Temp for unweighted graphs
        if w_final[i] == 1.0:
            G_combined.add_edge(u,v,weight=a['weight'])
        i = i + 1

    print("Computing new algebraic connectivity")
    start = timer()
    algconn = ac.algebraic_connectivity(G_combined, method='lanczos')
    end = timer()
    print("Success! elapsed time: ", (end - start))
    print("New alg. Conn: ", algconn)
