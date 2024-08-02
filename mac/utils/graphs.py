"""
Types and utilities for working with graphs.
"""

import numpy as np
from typing import List, Union, Tuple
from collections import namedtuple
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix

# Define Edge container
Edge = namedtuple("Edge", ["i", "j", "weight"])

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
    edges: List[Tuple[int, int]], weights: List[int], num_nodes: int
) -> csr_matrix:
    """Returns the weighted graph Laplacian matrix from a list
    of edges and edge weights

    Args:
        edges (List[Tuple[int, int]]): the list of edge indices (i,j)
        weights (List[float]): the list of edge weights
        num_nodes (int): the number of nodes

    Returns:
        csr_matrix: the weighted graph Laplacian matrix
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

    return csr_matrix(coo_matrix((data, (rows, cols)), shape=[num_nodes, num_nodes]))


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
