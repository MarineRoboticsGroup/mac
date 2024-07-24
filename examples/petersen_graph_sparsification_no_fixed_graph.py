import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx
from mac.baseline import NaiveGreedy
from mac.greedy_eig import GreedyEig
from mac.greedy_esp import GreedyESP
from mac.greedy_esp_minimal import MinimalGreedyESP
from mac.mac import MAC

plt.rcParams['text.usetex'] = True

G = nx.petersen_graph()
n = len(G.nodes())

# Add a chain
for i in range(n-1):
    if G.has_edge(i+1, i):
        G.remove_edge(i+1, i)
    if not G.has_edge(i, i+1):
        G.add_edge(i, i+1)

print(G)
pos = nx.shell_layout(G, nlist=[range(5,10), range(5)])
nx.draw(G, pos=pos)
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

edges = nx_to_mac(G)

# Split chain and non-chain parts
fixed_meas, candidate_meas = split_edges(edges)
edges = fixed_meas + candidate_meas

pct_candidates = 0.4
num_candidates = int(pct_candidates * len(candidate_meas))
mac = MAC([], edges, n)
greedy_eig = GreedyEig(fixed_meas, candidate_meas, n)
greedy_esp = GreedyESP(fixed_meas, candidate_meas, n)

w_init = np.zeros(len(fixed_meas) + len(candidate_meas))
w_init[:(len(fixed_meas) + num_candidates)] = 1.0

result, unrounded, upper = mac.fw_subset(w_init, len(fixed_meas) + num_candidates, max_iters=200, verbose=True, rounding="nearest")
#greedy_eig_result, _ = greedy_eig.subset(num_candidates)
#greedy_esp_result, _ = greedy_esp.subset(num_candidates)

init_selected = select_edges(edges, w_init)
init_selected_G = mac_to_nx(init_selected)

# greedy_eig_selected = select_edges(candidate_meas, greedy_eig_result)
# greedy_eig_selected_G = mac_to_nx(fixed_meas + greedy_eig_selected)

# greedy_esp_selected = select_edges(candidate_meas, greedy_esp_result)
#greedy_esp_selected_G = mac_to_nx(fixed_meas + greedy_esp_selected)

print(len(result))
print(len(edges))
selected = select_edges(edges, result)
selected_G = mac_to_nx(selected)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(131)
nx.draw(G, pos=pos)
plt.title(rf"Original ($\lambda_2$ = {mac.evaluate_objective(np.ones(len(w_init))):.3f})")
plt.subplot(132)
nx.draw(init_selected_G, pos=pos)
plt.title(rf"Naive ($\lambda_2$ = {mac.evaluate_objective(w_init):.3f})")
# plt.subplot(153)
# nx.draw(greedy_eig_selected_G, pos=pos)
# plt.title(rf"GreedyEig ($\lambda_2$ = {mac.evaluate_objective(greedy_eig_result):.3f})")
# plt.subplot(154)
# nx.draw(greedy_esp_selected_G, pos=pos)
# plt.title(rf"GreedyESP ($\lambda_2$ = {mac.evaluate_objective(greedy_esp_result):.3f})")
plt.subplot(133)
nx.draw(selected_G, pos=pos)
print(len(init_selected))
print(len(selected))
plt.title(rf"Ours ($\lambda_2$ = {mac.evaluate_objective(result):.3f})")
plt.show()

