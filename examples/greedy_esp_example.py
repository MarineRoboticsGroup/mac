import networkx as nx
from mac.mac import MAC
from mac.greedy_esp import GreedyESP
from mac.utils import split_edges, nx_to_mac, mac_to_nx, get_edge_selection_as_binary_mask
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":
    # set up graph in expected format
    n = 200
    p = 0.30
    G = nx.erdos_renyi_graph(n, p, seed=42)

    # Add a chain
    for i in range(n-1):
        if G.has_edge(i+1, i):
            G.remove_edge(i+1, i)
        if not G.has_edge(i, i+1):
            G.add_edge(i, i+1)

    edges = nx_to_mac(G)
    fixed_edges, candidate_edges = split_edges(edges)
    print(f"Graph has {len(edges)} edges")
    print(f"Fixed edges: {len(fixed_edges)}, candidate edges: {len(candidate_edges)}")
    mac = MAC(fixed_edges, candidate_edges, n)

    #  build the problem
    time_start = time.perf_counter()
    edge_selection_problem = GreedyESP(fixed_edges, candidate_edges, n)

    # solve the problem
    num_edges_to_select = 100
    selected_edges_mask, selected_edges = edge_selection_problem.subset(num_edges_to_select)

    print(f"Time taken: {time.perf_counter() - time_start}")

    # check that the solution is valid
    assert len(selected_edges) == num_edges_to_select
    assert len(set(selected_edges).intersection(set(fixed_edges))) == 0
    assert len(set(selected_edges).intersection(set(candidate_edges))) == num_edges_to_select

    # print the value of the objective function
    print(f"lambda2 GreedyESP: {mac.evaluate_objective(selected_edges_mask)}")

    # plot the solution
    all_edges_in_sparsified_graph = fixed_edges + selected_edges
    selected_graph = mac_to_nx(all_edges_in_sparsified_graph)
    plt.figure()
    nx.draw(selected_graph, with_labels=True)
    plt.show()
