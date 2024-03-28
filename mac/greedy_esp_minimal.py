import numpy as np
from typing import List, Tuple
from mac.utils import weight_graph_lap_from_edge_list, weight_graph_lap_from_edges, Edge

def incidence_vector(eij, num_nodes: int):
    incidence_vec = np.zeros(num_nodes)
    i = eij[0]
    j = eij[1]

    incidence_vec[i] = 1
    incidence_vec[j] = -1
    return incidence_vec

class MinimalGreedyESP:
    def __init__(self, fixed_edges: List[Edge], candidate_edges: List[Edge], num_poses, use_reduced_laplacian=False):
        self.L_odom = weight_graph_lap_from_edge_list(fixed_edges, num_poses)
        self.num_poses = num_poses
        self.laplacian_e_list = []
        self.edge_list = np.array([(e[0], e[1]) for e in candidate_edges])
        self.weights = np.array([e[2] for e in candidate_edges])
        self.use_reduced_laplacian = use_reduced_laplacian

        for edge in candidate_edges:
            laplacian_e = weight_graph_lap_from_edge_list([edge], num_poses)
            self.laplacian_e_list.append(laplacian_e)

        self.laplacian_e_list = np.array(self.laplacian_e_list)

    def combined_laplacian(self, w, tol=1e-10):
        """
        Construct the combined Laplacian (fixed edges plus candidate edges weighted by w).

        w: An element of [0,1]^m; this is the edge selection to use
        tol: Tolerance for edges that are numerically zero. This improves speed
        in situations where edges are not *exactly* zero, but close enough that
        they have almost no influence on the graph.

        returns the matrix L(w)
        """
        idx = np.where(w > tol)
        prod = w[idx]*self.weights[idx]
        C1 = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_poses)
        C = self.L_odom + C1
        return C

    def subset(self, k: int) -> Tuple[np.ndarray, List[Edge]]:
        """
        Apply GreedyEsp to select k edges from the candidate edges.

        Args:
            k (int): the number of edges to select

        Returns:
            np.ndarray: the selected edges as a boolean vector
            List[Edge]: the selected edges

        """
        solution = np.zeros(len(self.weights))
        selected_edges: List[Edge] = []

        for _ in range(k):
            # Placeholders to keep track of best measurement
            best_idx = -1
            best_weighted_reff = 0
            best_edge = None
            # Loop over all unselected measurements to find new best
            # measurement to add
            for j in range(len(self.weights)):
                # If measurement j is already selected, skip it
                if solution[j] > 0:
                    continue
                # Test solution
                w = np.copy(solution)
                curr_edge_ij = self.edge_list[j]
                auv = incidence_vector(curr_edge_ij, self.num_poses).reshape((self.num_poses, 1))

                reff = self.compute_reff(w, auv)
                weighted_reff = self.weights[j] * reff

                # add a numerical tolerance for the comparison so we
                # deterministically tie-break weighted effective resistances by
                # selecting the first edge with the max weighted reff. This was
                # actually a reason why some of the tests were failing
                tol = 1e-8
                if weighted_reff > best_weighted_reff + tol:
                    best_idx = j
                    best_weighted_reff = weighted_reff
                    best_edge = Edge(curr_edge_ij[0], curr_edge_ij[1], self.weights[j])
            # If best_idx is still -1, something went terribly wrong, or there
            # are no measurements
            assert(best_idx != -1)
            assert best_edge is not None
            solution[best_idx] = 1
            selected_edges.append(best_edge)
        return solution, selected_edges

    def compute_reff(self, w: np.ndarray, auv: np.ndarray):
        """
        Compute the effective resistance of the selected edges
        """
        L = self.combined_laplacian(w)
        if self.use_reduced_laplacian:
            auv = auv[1:]
            Linv = np.linalg.inv(L.todense()[1:,1:])
        else:
            Linv = np.linalg.pinv(L.todense())
        reff = (auv.T @ Linv @ auv)[0,0]
        return reff
