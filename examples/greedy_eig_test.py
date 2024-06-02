import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_edges, split_edges, Edge, nx_to_mac, mac_to_nx
from mac.baseline import NaiveGreedy
from mac.greedy_eig import GreedyEig
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
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

measurements = nx_to_mac(G)

# Split chain and non-chain parts
fixed_meas, candidate_meas = split_edges(measurements)

pct_candidates = 0.1
num_candidates = int(pct_candidates * len(candidate_meas))
mac = MAC(fixed_meas, candidate_meas, n)
naive_greedy_eig = GreedyEig(fixed_meas, candidate_meas, n)

w_init = np.zeros(len(candidate_meas))
w_init[:num_candidates] = 1.0

result, rounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=200)
init_selected = select_edges(candidate_meas, w_init)
selected = select_edges(candidate_meas, result)

init_selected_G = mac_to_nx(fixed_meas + init_selected)
selected_G = mac_to_nx(fixed_meas + selected)

w_greedy_eig, selected_eig = naive_greedy_eig.subset(num_candidates)
selected_G_eig = mac_to_nx(fixed_meas + selected_eig)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Greedy: {mac.evaluate_objective(w_greedy_eig)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(131)
nx.draw(init_selected_G)
plt.subplot(132)
nx.draw(selected_G_eig)
plt.subplot(133)
nx.draw(selected_G)
plt.show()

