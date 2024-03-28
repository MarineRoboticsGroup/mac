import numpy as np
import networkx as nx
from typing import List, Tuple

from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx, Edge

def get_split_petersen_graph() -> Tuple[List[Edge], List[Edge], int]:
    """
    Get a "split" Petersen graph for testing purposes.
    Returns:
        fixed_edges: a chain of edges in the graph.
        candidate_edges: the remaining edges in the graph.
        n: the number of nodes in the graph.
    """
    G = nx.petersen_graph()
    n = len(G.nodes())

    # Add a chain
    for i in range(n-1):
        if G.has_edge(i+1, i):
            G.remove_edge(i+1, i)
            pass
        if not G.has_edge(i, i+1):
            G.add_edge(i, i+1)
            pass
        pass

    edges = nx_to_mac(G)

    # Split chain and non-chain parts
    fixed_edges , candidate_edges = split_edges(edges)

    return fixed_edges, candidate_edges, n

def get_split_erdos_renyi_graph(n: int=20, p: float = 0.30) -> Tuple[List[Edge], List[Edge], int]:
    G = nx.erdos_renyi_graph(n, p)

    for i in range(n-1):
        if G.has_edge(i+1, i):
            G.remove_edge(i+1, i)
        if not G.has_edge(i, i+1):
            G.add_edge(i, i+1)

    edges = nx_to_mac(G)
    fixed_edges, candidate_edges = split_edges(edges)
    return fixed_edges, candidate_edges, n