import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
import networkx as nx
import math
from typing import List, Union, Tuple

# Define Edge container
Edge = namedtuple("Edge", ["i", "j", "weight"])

def round_nearest(w, k, weights=None, break_ties_decimal_tol=None):
    """
    Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to a
    solution to the original problem with w_i \in {0,1}. Ties between edges
    are broken based on the original weight (all else being equal, we
    prefer edges with larger weight).

    w: A solution in the feasible set for the relaxed problem
    weights: The original weights of the edges to use as a tiebreaker
    k: The number of edges to select
    break_ties_decimal_tol: tolerance for determining floating point equality of weights w. If two selection weights are equal to this many decimal places, we break the tie based on the original weight.

    returns w': A solution in the feasible set for the original problem
    """
    if weights is None or break_ties_decimal_tol is None:
        # If there are no tiebreakers, just set the top k elements of w to 1,
        # and the rest to 0
        idx = np.argpartition(w, -k)[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    # If there are tiebreakers, we truncate the selection weights to the
    # specified number of decimal places, and then break ties based on the
    # original weights
    truncated_w = w.round(decimals=break_ties_decimal_tol)
    zipped_vals = np.array(
        [(truncated_w[i], weights[i]) for i in range(len(w))],
        dtype=[("w", "float"), ("weight", "float")],
    )
    idx = np.argpartition(zipped_vals, -k, order=["w", "weight"])[-k:]
    rounded = np.zeros(len(w))
    if k > 0:
        rounded[idx] = 1.0
    return rounded

def round_random(w, k):
    """
    Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to
    one with hard edge constraints and satisfying the constraint that the
    expected number of selected edges is equal to k.

    w: A solution in the feasible set for the relaxed problem
    k: The number of edges to select _in expectation_

    returns w': A solution containing hard edge selections with an expected
    number of selected edges equal to k.
    """
    x = np.zeros(len(w))
    for i in range(len(w)):
        r = np.random.rand()
        if w[i] > r:
            x[i] = 1.0
    return x

def round_madow(w, k, seed=None, value_fn=None, max_iters=1):
    if value_fn is None or max_iters == 1:
        return round_madow_base(w, k, seed)

    best_x = None
    best_val = -np.inf
    for i in range(max_iters):
        x = round_madow_base(w, k, seed)
        val = value_fn(x)
        if val > best_val:
            best_val = val
            best_x = x
    return best_x


def round_madow_base(w, k, seed=None):
    """
    Use Madow rounding
    """
    if seed is None:
        u = np.random.rand()
    else:
        u = seed.rand()
    x = np.zeros(len(w))
    # pi = np.zeros(len(w) + 1)
    pi = np.zeros(len(w))
    sumw = np.cumsum(w)
    pi[1:] = sumw[:-1]
    for i in range(k):
        total = u + i
        x[np.where((pi <= total) & (total < sumw))] = 1.0
        # for j in range(len(w)):
        #     if (x[j] != 1) and (pi[j] <= total) and (total < pi[j+1]):
        #         x[j] = 1.0
        #         break

    assert np.sum(x) == k, f"Error: {np.sum(x)} != {k}"
    return x

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


def weight_graph_lap_from_edge_list(edges: List[Edge], num_nodes: int) -> csr_matrix:
    """Returns the (sparse) weighted graph Laplacian matrix from the list of edges

    Args:
        edges (List[Edge]): the list of edges
        num_nodes (int): the number of variables

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

    return csr_matrix(coo_matrix((data, (rows, cols)), shape=[num_nodes, num_nodes]))


def weight_reduced_graph_lap_from_edge_list(
    edges: List[Edge], num_nodes: int
) -> csr_matrix:
    graph_lap = weight_graph_lap_from_edge_list(edges, num_nodes)
    return graph_lap[1:, 1:]


def weight_graph_lap_from_edges(
    edges: List[Tuple[int, int]], weights: List[int], num_poses: int
) -> csr_matrix:
    """Returns the sparse rotational weighted graph Laplacian matrix from a list
    of edges and edge weights

    Args:
        edges (List[Tuple[int, int]]): the list of edge indices (i,j)
        weights (List[float]): the list of edge weights
        num_poses (int): the number of poses

    Returns:
        csr_matrix: the rotational weighted graph Laplacian matrix
    """
    assert len(edges) == len(weights)
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


def split_edges(edges: List[Edge]) -> Tuple[List[Edge], List[Edge]]:
    """Splits list of edges into a "fixed" chain part and a set of candidate
    loops.

    Args:
        edges: List of edges.

    Returns:
        A tuple containing two lists:
            fixed: edges where |i - j| = 1, and
            candidates: edges where |i - j| != 1

    This is particularly useful for pose-graph SLAM applications where the
    fixed edges correspond to an odometry chain and the candidate edges
    correspond to loop closures.

    """
    chain_edges = []
    loop_edges = []
    for edge in edges:
        id1 = edge.i
        id2 = edge.j
        if abs(id2 - id1) > 1:
            loop_edges.append(edge)
        else:
            chain_edges.append(edge)

    return chain_edges, loop_edges


def select_edges(edges, w):
    """
    Select the subset of edges from `edges` with weight equal to one in
    `w`.
    """
    assert len(edges) == len(w), f"Selection mask length {len(w)} does not match number of edges {len(edges)}"
    edges_out = []
    for i, edge in enumerate(edges):
        if w[i] == 1.0:
            edges_out.append(edge)
    return edges_out


def get_incidence_vector(eij: Union[Edge, Tuple[int, int]], num_nodes: int):
    """Returns the incidence vector for the edge eij

    Args:
        eij (Union[Edge, Tuple[int, int]]): the edge
        num_nodes (int): the number of nodes

    Returns:
        np.ndarray: the incidence vector
    """
    incidence_vec = np.zeros(num_nodes)
    i = eij[0]
    j = eij[1]

    incidence_vec[i] = 1
    incidence_vec[j] = -1
    return incidence_vec


def set_incidence_vector_for_edge_inplace(
    auv_vec: np.ndarray, edge: Union[Edge, Tuple[int, int]], num_nodes: int
) -> None:
    """Modifies the passed in auv vector to be the correct values for the
    given edge.

    NOTE: Assumes that the edge indices are in the range [0, num_nodes) and
    that we are getting the incidence vector for a reduced Laplacian
    (i.e. indices are shifted by -1).

    Args:
        auv_vec (np.ndarray): the auv vector to modify
        edge (Union[Edge, Tuple[int, int]]): the edge
        num_nodes (int): the number of nodes

    """
    assert len(auv_vec) == num_nodes - 1
    auv_vec.fill(0)
    i = edge[0] - 1
    j = edge[1] - 1
    if i >= 0:
        auv_vec[i] = 1
    if j >= 0:
        auv_vec[j] = -1


def get_edge_selection_as_binary_mask(
    edges: List[Edge], selected_edges: List[Edge]
) -> np.ndarray:
    """
    Returns a binary mask of the selected edges.

    Args:
        edges: List of edges.
        selected_edges: List of selected edges.

    Returns:
        A binary mask of the selected edges.
    """
    assert len(edges) >= len(
        selected_edges
    ), "The number of selected edges cannot be greater than the total number of edges."
    mask = np.zeros(len(edges))
    for i, edge in enumerate(edges):
        if edge in selected_edges:
            mask[i] = 1.0
    return mask

