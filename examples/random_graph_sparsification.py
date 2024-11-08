import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils.graphs import select_edges, Edge
from mac.utils.conversions import nx_to_mac, mac_to_nx
from mac.solvers import MAC

n = 20
p = 0.6
G = nx.erdos_renyi_graph(n, p)

# Add a chain
for i in range(n-1):
    if G.has_edge(i+1, i):
        G.remove_edge(i+1, i)
    if not G.has_edge(i, i+1):
        G.add_edge(i, i+1)

# Split the graph into a tree part and a "loop" part
spanning_tree = nx.minimum_spanning_tree(G)
loop_graph = nx.difference(G, spanning_tree)

nx.draw(G)
plt.title("Original Graph")
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

fixed_edges = nx_to_mac(spanning_tree)
candidate_edges = nx_to_mac(loop_graph)

pct_candidates = 0.2
num_candidates = int(pct_candidates * len(candidate_edges))
mac = MAC(fixed_edges, candidate_edges, n)

w_init = np.zeros(len(candidate_edges))
w_init[:num_candidates] = 1.0
np.random.shuffle(w_init)

result, rounded, upper = mac.solve(num_candidates, w_init, max_iters=100, rounding="madow", use_cache=True)
init_selected = select_edges(candidate_edges, w_init)
selected = select_edges(candidate_edges, result)

init_selected_G = mac_to_nx(fixed_edges + init_selected)
selected_G = mac_to_nx(fixed_edges + selected)

print(f"lambda2 Initial: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(121)
nx.draw(init_selected_G)
plt.title(f"Initial Selection\n$\lambda_2$ = {mac.evaluate_objective(w_init):.2f}")
plt.subplot(122)
nx.draw(selected_G)
plt.title(f"Ours\n$\lambda_2$ = {mac.evaluate_objective(result):.2f}")
plt.show()

