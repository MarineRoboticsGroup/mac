import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_edges, split_edges, Edge, nx_to_mac, mac_to_nx
from mac.baseline import NaiveGreedy
from mac.mac import MAC

n = 20
p = 0.99
G = nx.erdos_renyi_graph(n, p)

# Add a chain
for i in range(n-1):
    if G.has_edge(i+1, i):
        G.remove_edge(i+1, i)
    if not G.has_edge(i, i+1):
        G.add_edge(i, i+1)

print(G)
nx.draw(G)
plt.title("Original Graph")
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

edges = nx_to_mac(G)

# Split chain and non-chain parts
fixed_edges, candidate_edges = split_edges(edges)

pct_candidates = 0.1
num_candidates = int(pct_candidates * len(candidate_edges))
mac = MAC(fixed_edges, candidate_edges, n)

w_init = np.zeros(len(candidate_edges))
w_init[:num_candidates] = 1.0

result, rounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=50)
init_selected = select_edges(candidate_edges, w_init)
selected = select_edges(candidate_edges, result)

init_selected_G = mac_to_nx(fixed_edges + init_selected)
selected_G = mac_to_nx(fixed_edges + selected)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(121)
nx.draw(init_selected_G)
plt.title(f"Random Selection\n$\lambda_2$ = {mac.evaluate_objective(w_init):.2f}")
plt.subplot(122)
nx.draw(selected_G)
plt.title(f"Ours\n$\lambda_2$ = {mac.evaluate_objective(result):.2f}")
plt.show()

