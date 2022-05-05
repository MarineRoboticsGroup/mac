import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_measurements, split_measurements, nx_to_mac, mac_to_nx
from mac.baseline import NaiveGreedy
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

measurements = nx_to_mac(G)

# Split chain and non-chain parts
fixed_meas, candidate_meas = split_measurements(measurements)

pct_candidates = 0.3
num_candidates = int(pct_candidates * len(candidate_meas))
mac = MAC(fixed_meas, candidate_meas, n)

w_init = np.zeros(len(candidate_meas))
w_init[:num_candidates] = 1.0

result, rounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=50)
init_selected = select_measurements(candidate_meas, w_init)
selected = select_measurements(candidate_meas, result)

init_selected_G = mac_to_nx(fixed_meas + init_selected)
selected_G = mac_to_nx(fixed_meas + selected)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(131)
nx.draw(G, pos=pos)
plt.title(rf"Original ($\lambda_2$ = {mac.evaluate_objective(np.ones(len(w_init))):.3f})")
plt.subplot(132)
nx.draw(init_selected_G, pos=pos)
plt.title(rf"Naive ($\lambda_2$ = {mac.evaluate_objective(w_init):.3f})")
plt.subplot(133)
nx.draw(selected_G, pos=pos)
plt.title(rf"Ours ($\lambda_2$ = {mac.evaluate_objective(result):.3f})")
plt.show()

