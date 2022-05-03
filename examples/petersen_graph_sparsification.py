import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mac.utils import select_measurements, split_measurements, RelativePoseMeasurement
from mac.baseline import NaiveGreedy
from mac.mac import MAC

def nx_to_rpm(G):
    measurements = []
    for edge in G.edges():
        t = np.zeros(3)
        R = np.eye(3)
        meas = RelativePoseMeasurement(edge[0], edge[1], t, R, 1.0, 1.0)
        measurements.append(meas)
    return measurements


def rpm_to_nx(measurements):
    G = nx.Graph()
    for meas in measurements:
        G.add_edge(meas.i, meas.j, weight=meas.kappa)
    return G


G = nx.petersen_graph()
n = len(G.nodes())

# Add a chain
for i in range(n-1):
    if G.has_edge(i+1, i):
        G.remove_edge(i+1, i)
    if not G.has_edge(i, i+1):
        G.add_edge(i, i+1)

print(G)
pos = nx.shell_layout(G)
nx.draw(G, pos=pos)
plt.show()

# Ensure G is connected before proceeding
assert(nx.is_connected(G))

measurements = nx_to_rpm(G)

# Split chain and non-chain parts
fixed_meas, candidate_meas = split_measurements(measurements)

pct_candidates = 0.5
num_candidates = int(pct_candidates * len(candidate_meas))
mac = MAC(fixed_meas, candidate_meas, n)

w_init = np.zeros(len(candidate_meas))
w_init[:num_candidates] = 1.0

result, rounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=50)
init_selected = select_measurements(candidate_meas, w_init)
selected = select_measurements(candidate_meas, result)

init_selected_G = rpm_to_nx(fixed_meas + init_selected)
selected_G = rpm_to_nx(fixed_meas + selected)

print(f"lambda2 Random: {mac.evaluate_objective(w_init)}")
print(f"lambda2 Ours: {mac.evaluate_objective(result)}")

plt.subplot(121)
nx.draw(init_selected_G, pos=pos)
plt.subplot(122)
nx.draw(selected_G, pos=pos)
plt.show()

