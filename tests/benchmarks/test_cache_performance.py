import numpy as np
import networkx as nx
import pytest

# MAC requirements
from mac.solvers.mac import MAC
from mac.utils.conversions import nx_to_mac

@pytest.mark.parametrize("k", [5])
def test_benchmark_solver(benchmark, k):
    # Load graph
    G = nx.petersen_graph()
    n = len(G.nodes())

    # Split the graph into a tree part and a "loop" part
    spanning_tree = nx.minimum_spanning_tree(G)
    loop_graph = nx.difference(G, spanning_tree)

    # Create solver
    solver = MAC(fixed_edges=nx_to_mac(spanning_tree),
              candidate_edges=nx_to_mac(loop_graph), num_nodes=n)

    # Set up initial guess
    x_init = np.zeros(len(loop_graph.edges()))
    x_init[:k] = 1.0

    result = benchmark.pedantic(solver.solve, args=[k], kwargs={"x_init": x_init,
                                                                "use_cache": False}, rounds=5)

@pytest.mark.parametrize("k", [5])
def test_benchmark_solver_cache(benchmark, k):
    # Load graph
    G = nx.petersen_graph()
    n = len(G.nodes())

    # Split the graph into a tree part and a "loop" part
    spanning_tree = nx.minimum_spanning_tree(G)
    loop_graph = nx.difference(G, spanning_tree)

    # Create solver
    solver = MAC(fixed_edges=nx_to_mac(spanning_tree),
              candidate_edges=nx_to_mac(loop_graph), num_nodes=n)

    # Set up initial guess
    x_init = np.zeros(len(loop_graph.edges()))
    x_init[:k] = 1.0

    result = benchmark.pedantic(solver.solve, args=[k], kwargs={"x_init": x_init,
                                                                "use_cache": True}, rounds=5)
