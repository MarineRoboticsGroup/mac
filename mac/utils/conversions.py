"""
Utilities for converting between graph types.
"""
import networkx as nx
from typing import List

from .graphs import Edge

def nx_to_mac(G: nx.Graph) -> List[Edge]:
    """Returns the list of edges in the graph G

    Args:
        G (nx.Graph): the graph

    Returns:
        List[Edge]: the list of edges
    """
    edges = []
    for nxedge in G.edges():
        i = nxedge[0]
        j = nxedge[1]
        data = G.get_edge_data(i, j)
        weight = 1.0
        if "weight" in data:
            weight = data["weight"]
        if i < j:
            edge = Edge(i, j, weight)
        else:
            edge = Edge(j, i, weight)
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
        if edge.i < edge.j:
            G.add_edge(edge.i, edge.j, weight=edge.weight)
        else:
            G.add_edge(edge.j, edge.i, weight=edge.weight)
    return G
