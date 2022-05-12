import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
import networkx as nx
from typing import List

import numba
from numba import jit

# Define Edge container
Edge = namedtuple('Edge', ['i', 'j', 'weight'])

def nx_to_mac(G: nx.Graph) -> List[Edge]:
    """Returns the list of edges in the graph G

    Args:
        G (nx.Graph): the graph

    Returns:
        List[Edge]: the list of edges
    """
    edges = []
    for nxedge in G.edges():
        edge = Edge(nxedge[0], nxedge[1], 1.0)
        edges.append(edge)
    return edges

def mac_to_nx(edges: List[Edge]) -> nx.Graph:
    """returns the graph corresponding to the list of edges

    Args:
        edges (List[Edge]): the list of edges

    Returns:
        nx.Graph: the networkx graph
    """
    G = nx.Graph()
    for edge in edges:
        G.add_edge(edge.i, edge.j, weight=edge.weight)
    return G

def weight_graph_lap_from_edge_list(edges: List[Edge], num_vars: int) -> csr_matrix:
    """Returns the (sparse) weighted graph Laplacian matrix from the list of edges

    Args:
        edges (List[Edge]): the list of edges
        num_vars (int): the number of variables

    Returns:
        csr_matrix: the weighted graph Laplacian matrix
    """
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

def weight_graph_lap_from_edges(edges: List[Edge], weights: List[int], num_poses: int) -> csr_matrix:
    """Returns the sparse rotational weighted graph Laplacian matrix from a list
    of edges and edge weights

    Args:
        edges (List[Edge]): the list of edges
        weights (List[float]): the list of edge weights
        num_poses (int): the number of poses

    Returns:
        csr_matrix: the rotational weighted graph Laplacian matrix
    """
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
    """
    Select the subset of edges from `measurements` with weight equal to one in
    `w`.
    """
    assert(len(measurements) == len(w))
    meas_out = []
    for i, meas in enumerate(measurements):
        if w[i] == 1.0:
            meas_out.append(meas)
    return meas_out
