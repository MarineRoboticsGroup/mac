import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
import networkx as nx

import numba
from numba import jit

# Define Edge container
Edge = namedtuple('Edge', ['i', 'j', 'weight'])

def nx_to_mac(G):
    edges = []
    for nxedge in G.edges():
        edge = Edge(nxedge[0], nxedge[1], 1.0)
        edges.append(edge)
    return edges

def mac_to_nx(edges):
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge.i, edge.j, weight=edge.weight)
    return G

def weight_graph_lap_from_edge_list(edges, num_vars):
    # Preallocate triplets
    rows = []
    cols = []
    data = []
    for edge in edges:
        # Diagonal elem (u,u)
        rows.append(edge.i)
        cols.append(edge.i)
        data.append(edge.weight)

        # Diagonal elem (v,v)
        rows.append(edge.j)
        cols.append(edge.j)
        data.append(edge.weight)

        # Off diagonal (u,v)
        rows.append(edge.i)
        cols.append(edge.j)
        data.append(-edge.weight)

        # Off diagonal (v,u)
        rows.append(edge.j)
        cols.append(edge.i)
        data.append(-edge.weight)

    return csr_matrix(coo_matrix((data, (rows, cols)), shape=[num_vars, num_vars]))

def rotational_weight_graph_lap_from_meas(measurements, num_poses):
    # Preallocate triplets
    rows = []
    cols = []
    data = []
    for meas in measurements:
        # Diagonal elem (u,u)
        rows.append(meas.i)
        cols.append(meas.i)
        data.append(meas.kappa)

        # Diagonal elem (v,v)
        rows.append(meas.j)
        cols.append(meas.j)
        data.append(meas.kappa)

        # Off diagonal (u,v)
        rows.append(meas.i)
        cols.append(meas.j)
        data.append(-meas.kappa)

        # Off diagonal (v,u)
        rows.append(meas.j)
        cols.append(meas.i)
        data.append(-meas.kappa)

    return csr_matrix(coo_matrix((data, (rows, cols)), shape=[num_poses, num_poses]))

def weight_graph_lap_from_edges(edges, weights, num_poses):
    # Preallocate triplets
    rows = []
    cols = []
    data = []
    for i in range(len(edges)):
        # Diagonal elem (u,u)
        rows.append(edges[i, 0])
        cols.append(edges[i, 0])
        data.append(weights[i])

        # Diagonal elem (v,v)
        rows.append(edges[i, 1])
        cols.append(edges[i, 1])
        data.append(weights[i])

        # Off diagonal (u,v)
        rows.append(edges[i, 0])
        cols.append(edges[i, 1])
        data.append(-weights[i])

        # Off diagonal (v,u)
        rows.append(edges[i, 1])
        cols.append(edges[i, 0])
        data.append(-weights[i])

    return csr_matrix(coo_matrix((data, (rows, cols)), shape=[num_poses, num_poses]))

def split_measurements(measurements):
    """
    Splits list of "measurements" and returns two lists:
    "odometry" - measurements where |i - j| = 1, and
    "loop closures" - measurements where |i - j| != 1
    """
    odom_measurements = []
    lc_measurements = []
    for measurement in measurements:
        id1 = measurement.i
        id2 = measurement.j
        if abs(id2 - id1) > 1:
            lc_measurements.append(measurement)
        else:
            odom_measurements.append(measurement)

    return odom_measurements, lc_measurements

def select_measurements(measurements, w):
    assert(len(measurements) == len(w))
    meas_out = []
    for i, meas in enumerate(measurements):
        if w[i] == 1.0:
            meas_out.append(meas)
    return meas_out
