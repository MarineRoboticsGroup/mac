"""
Utilities for converting between graph types.
"""
import networkx as nx
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
