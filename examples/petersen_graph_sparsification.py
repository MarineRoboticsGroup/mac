import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx
from mac.baseline import NaiveGreedy
from mac.greedy_eig import GreedyEig
from mac.greedy_esp import GreedyESP
from mac.mac import MAC

plt.rcParams['text.usetex'] = True

G = nx.petersen_graph()
n = len(G.nodes())

print(G)
pos = nx.shell_layout(G, nlist=[range(5,10), range(5)], rotate=np.pi/32)
nx.draw(G, pos=pos)
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

# Split the graph into a tree part and a "loop" part
spanning_tree = nx.minimum_spanning_tree(G)
loop_graph = nx.difference(G, spanning_tree)

nx.draw(spanning_tree, pos=pos)
plt.show()

fixed_edges = nx_to_mac(spanning_tree)
candidate_edges = nx_to_mac(loop_graph)

pct_candidates = 0.4
num_candidates = int(pct_candidates * len(candidate_edges))
mac = MAC(fixed_edges, candidate_edges, n)
greedy_eig = GreedyEig(fixed_edges, candidate_edges, n)
greedy_esp = GreedyESP(fixed_edges, candidate_edges, n)

w_init = np.zeros(len(candidate_edges))
w_init[:num_candidates] = 1.0

result, unrounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=100, rounding="nearest")
greedy_eig_result, _ = greedy_eig.subset(num_candidates)
greedy_esp_result, _ = greedy_esp.subset(num_candidates)

init_selected = select_edges(candidate_edges, w_init)
init_selected_G = mac_to_nx(fixed_edges + init_selected)

greedy_eig_selected = select_edges(candidate_edges, greedy_eig_result)
greedy_eig_selected_G = mac_to_nx(fixed_edges + greedy_eig_selected)

greedy_esp_selected = select_edges(candidate_edges, greedy_esp_result)
greedy_esp_selected_G = mac_to_nx(fixed_edges + greedy_esp_selected)

selected = select_edges(candidate_edges, result)
selected_G = mac_to_nx(fixed_edges + selected)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(151)
nx.draw(G, pos=pos)
plt.title(rf"Original ($\lambda_2$ = {mac.evaluate_objective(np.ones(len(w_init))):.3f})")
plt.subplot(152)
nx.draw(init_selected_G, pos=pos)
plt.title(rf"Naive ($\lambda_2$ = {mac.evaluate_objective(w_init):.3f})")
plt.subplot(153)
nx.draw(greedy_eig_selected_G, pos=pos)
plt.title(rf"GreedyEig ($\lambda_2$ = {mac.evaluate_objective(greedy_eig_result):.3f})")
plt.subplot(154)
nx.draw(greedy_esp_selected_G, pos=pos)
plt.title(rf"GreedyESP ($\lambda_2$ = {mac.evaluate_objective(greedy_esp_result):.3f})")
plt.subplot(155)
nx.draw(selected_G, pos=pos)
plt.title(rf"Ours ($\lambda_2$ = {mac.evaluate_objective(result):.3f})")
plt.show()

