from mac.utils import *
import numpy as np
import networkx as nx
import networkx.linalg as la
import networkx.generators as gen
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg

from timeit import default_timer as timer

from collections import namedtuple

MACResult = namedtuple('MACResult', ['w', 'F_unrounded', 'objective_values', 'duality_gaps'])

class MAC:
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
        assert(method != 'lobpcg') # LOBPCG not supported at the moment
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

    def evaluate_fiedler_pair(self, w, method='tracemin_lu', tol=1e-8):
        """
        Compute the second smallest eigenvalue of L(w) and corresponding
        eigenvector using `method` and tolerance `tol`. This is just a helper
        that constructs L(w) and calls `self.find_fiedler_pair` on the
        resulting matrix, passing along the arguments.

        w: An element of [0,1]^m; this is the edge selection to use
        method: Any method supported by NetworkX for computing algebraic
        connectivity. See:
        https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html

        tol: Numerical tolerance for eigenvalue computation

        returns a tuple (lambda_2(L(w)), v_2(L(w))) containing the Fiedler
        value and corresponding vector.

        """
        return self.find_fiedler_pair(self.combined_laplacian(w), method, tol)

    def evaluate_objective(self, w):
        """
        Compute lambda_2(L(w)) where L(w) is the Laplacian with edge i weighted
        by w_i and lambda_2 is the second smallest eigenvalue (this is the
        algebraic connectivity).

        w: Weights for each edge

        returns F(w) = lambda_2(L(w)).
        """
        L = self.combined_laplacian(w)
        return self.find_fiedler_pair(L)[0]

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

    def round_solution(self, w, k):
        """
        Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to a
        solution to the original problem with w_i \in {0,1}. Ties between edges
        are broken arbitrarily.

        w: A solution in the feasible set for the relaxed problem
        k: The number of edges to select

        returns w': A solution in the feasible set for the original problem
        """
        idx = np.argpartition(w, -k)[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    def simple_random_round(self, w, k):
        """
        Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to
        one with hard edge constraints and satisfying the constraint that the
        expected number of selected edges is equal to k.

        w: A solution in the feasible set for the relaxed problem
        k: The number of edges to select _in expectation_

        returns w': A solution containing hard edge selections with an expected
        number of selected edges equal to k.
        """
        x = np.zeros(len(w))
        for i in range(len(w)):
            r = np.random.rand()
            if w[i] > r:
                x[i] = 1.0
        return x

    def round_solution_tiebreaker(self, w, k, decimal_tol=10):
        """
        Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to a
        solution to the original problem with w_i \in {0,1}. Ties between edges
        are broken based on the original weight (all else being equal, we
        prefer edges with larger weight).

        w: A solution in the feasible set for the relaxed problem
        k: The number of edges to select
        decimal_tol: tolerance for determining floating point equality of weights w

        returns w': A solution in the feasible set for the original problem
        """
        truncated_w = w.round(decimals=decimal_tol)
        zipped_vals = np.array([(truncated_w[i], self.weights[i]) for i in range(len(w))], dtype=[('w', 'float'), ('weight', 'float')])
        idx = np.argpartition(zipped_vals, -k, order=['w', 'weight'])[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    def fw_subset(self, w_init, k, max_iters=5, duality_gap_tol=1e-8):
        """
        Use the Frank-Wolfe method to solve the subset selection problem,.

        w_init: Array-like, must satisfy 0 <= w_i <= 1, |w| <= k
        k: size of max allowed subset
        max_iters: Maximum number of Frank-Wolfe iterations before returning
        duality_gap_tol: Minimum duality gap (for early stopping)

        returns a tuple (solution, unrounded, upper_bound) where
        solution: the (rounded) solution w \in {0,1}^m |w| = k
        unrounded: the solution obtained prior to rounding
        upper_bound: the value of the dual at the last iteration
        """
        u_i = float("inf")
        w_i = w_init
        zeros = np.zeros(len(w_init))
        obj_prev = None
        for it in range(max_iters):
            # Compute gradient
            f_i, vec_i = self.evaluate_fiedler_pair(w_i)
            grad_i = self.grad_from_fiedler(vec_i)

            # Solve the direction-finding subproblem by maximizing the linear
            # approximation of f at w_i
            s_i = self.round_solution(grad_i, k)

            # Compute dual upper bound from linear approximation
            # f_i = self.evaluate_objective(w_i)
            u_i = min(u_i, f_i + grad_i @ (s_i - w_i))

            # If the duality gap is sufficiently small, we are done
            if u_i - f_i < duality_gap_tol:
                print("Duality gap tolerance reached, found optimal solution")
                return self.round_solution_tiebreaker(w_i, k), w_i, u_i
                # return self.simple_random_round(w_i, k), w_i

            # Step size determination - naive method. No line search
            alpha = 2.0 / (it + 2.0)
            w_i = w_i + alpha * (s_i - w_i)

        print("Reached maximum iterations")
        return self.round_solution_tiebreaker(w_i, k), w_i, u_i
