from mac.utils import *
import numpy as np

import networkx as nx
import networkx.linalg as la

class MinimalGreedyEig:
    def __init__(self, odom_measurements, lc_measurements, num_poses):
        self.L_odom = weight_graph_lap_from_edge_list(odom_measurements, num_poses)
        self.num_poses = num_poses
        self.laplacian_e_list = []
        self.weights = []
        self.edge_list = []

        for meas in lc_measurements:
            laplacian_e = weight_graph_lap_from_edge_list([meas], num_poses)
            self.laplacian_e_list.append(laplacian_e)
            self.weights.append(meas.weight)
            self.edge_list.append((meas.i,meas.j))

        self.laplacian_e_list = np.array(self.laplacian_e_list)
        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

    def find_fiedler_pair(self, L, method='tracemin_lu', tol=1e-8):
        """
        Compute the second smallest eigenvalue of L and corresponding
        eigenvector using `method` and tolerance `tol`.

        w: An element of [0,1]^m; this is the edge selection to use
        method: Any method supported by NetworkX for computing algebraic
        connectivity. See:
        https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html

        tol: Numerical tolerance for eigenvalue computation

        returns a tuple (lambda_2(L), v_2(L)) containing the Fiedler
        value and corresponding vector.

        """
        find_fiedler_func = la.algebraicconnectivity._get_fiedler_func(method)
        x = None
        output = find_fiedler_func(L, x=x, normalized=False, tol=tol, seed=np.random.RandomState(7))
        return output

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

    def grad_from_fiedler(self, fiedler_vec):
        """
        Compute a (super)gradient of the algebraic connectivity with respect to w
        from the Fiedler vector.

        fiedler_vec: Eigenvector of the Laplacian corresponding to the second eigenvalue.

        returns grad F(w) from equation (8) of our paper: https://arxiv.org/pdf/2203.13897.pdf.
        """
        grad = np.zeros(len(self.weights))

        for k in range(len(self.weights)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            grad[k] = kdelta * (v_i - v_j)
        return grad

    def subset(self, k, save_intermediate=False):
        # Initialize w to be all zeros
        solution = np.zeros(len(self.weights))

        # Compute the Fiedler vector and value for the fixed subgraph
        l2, fiedler_vec = self.find_fiedler_pair(self.L_odom)
        grad = self.grad_from_fiedler(fiedler_vec)
        solution_l2 = l2
        solution_grad = grad
        selected_edges = []
        for i in range(k):
            # Placeholders to keep track of best measurement
            best_idx = -1
            best_l2 = 0
            best_grad = np.zeros(len(self.weights))
            # Loop over all unselected measurements to find new best
            # measurement to add
            for j in range(len(self.weights)):
                # If measurement j is already selected, skip it
                if solution[j] > 0:
                    continue

                # Construct a test vector with edge j added
                w = np.copy(solution)
                w[j] = 1

                # Compute a linear approximation to L2(w) evaluated at the
                # current solution. The value of this linear approximation at
                # w[j]=1 gives an upper bound on L2(w) at this point. This
                # means that if u is less than the best l2, there's no way this
                # edge will be the best, and we can safely discard it.
                # u = best_l2 + solution_grad @ (w - solution)
                u = solution_l2 + solution_grad[j]
                if u < best_l2:
                    continue

                # If we made it here, we need to compute the true L2(w) and
                # gradient.
                L = self.combined_laplacian(w)
                l2, fiedler_vec = self.find_fiedler_pair(L)
                grad = self.grad_from_fiedler(fiedler_vec)

                # add a numerical tolerance for the comparison so we
                # deterministically tie-break weighted effective resistances by
                # selecting the first edge with the max weighted reff. This was
                # actually a reason why some of the tests were failing
                tol = 1e-8
                if l2 > best_l2 + tol:
                    best_idx = j
                    best_l2 = l2
                    best_grad = grad
            # If best_idx is still -1, something went terribly wrong, or there
            # are no measurements
            assert(best_idx != -1)
            solution[best_idx] = 1
            solution_l2 = best_l2
            solution_grad = best_grad
            best_edge_ij = self.edge_list[best_idx]
            best_edge = Edge(best_edge_ij[0], best_edge_ij[1], self.weights[best_idx])
            selected_edges.append(best_edge)

        return solution, selected_edges

