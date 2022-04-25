from utils import *
import numpy as np
import networkx as nx
import networkx.linalg as la
import networkx.generators as gen
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg

from timeit import default_timer as timer

from collections import namedtuple

FWACResult = namedtuple('FWACResult', ['w', 'F_unrounded', 'objective_values', 'duality_gaps'])

class FWAC:
    def __init__(self, odom_measurements, lc_measurements, num_poses):
        self.L_odom = rotational_weight_graph_lap_from_meas(odom_measurements, num_poses)
        self.num_poses = num_poses
        self.laplacian_e_list = []
        self.kappas = []
        self.edge_list = []

        for meas in lc_measurements:
            laplacian_e = rotational_weight_graph_lap_from_meas([meas], num_poses)
            self.laplacian_e_list.append(laplacian_e)
            self.kappas.append(meas.kappa)
            self.edge_list.append((meas.i,meas.j))

        self.laplacian_e_list = np.array(self.laplacian_e_list)
        self.kappas = np.array(self.kappas)
        self.edge_list = np.array(self.edge_list)

    def find_fiedler_pair(self, L, method='tracemin_lu', tol=1e-8):
        assert(method != 'lobpcg') # LOBPCG not supported at the moment
        # start = timer()
        find_fiedler_func = la.algebraicconnectivity._get_fiedler_func(method)
        x = None
        output = find_fiedler_func(L, x=x, normalized=False, tol=tol, seed=np.random.RandomState(7))
        # end = timer()
        # print("scipy Fiedler time: ", end - start)
        return output

    def combined_laplacian(self, w, tol=1e-10):
        start = timer()
        idx = np.where(w > tol)
        # print(idx)
        # C1 = sum(laplacian_e * weight for (weight, laplacian_e) in zip(w[idx], self.laplacian_e_list[idx]))
        # mult = w[idx] * self.laplacian_e_list[idx]
        prod = w[idx]*self.kappas[idx]
        end = timer()
        # print("mult time: ", end - start)
        start = timer()
        # C1 = sum_sparse_orig(self.laplacian_e_list[idx], w[idx])
        # C1 = sum_sparse_orig(mult)
        C1 = rotational_weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_poses)
        # end = timer()
        # print("Sparse sum time: ", end - start)
        # print("Rot lap build time: ", end - start)
        # start = timer()
        C = self.L_odom + C1
        # end = timer()
        # print("LAP C time: ", end - start)
        return C

    def evaluate_fiedler_pair(self, w, method='tracemin_lu', tol=1e-8):
        return self.find_fiedler_pair(self.combined_laplacian(w), method, tol)

    def evaluate_objective(self, w):
        L = self.combined_laplacian(w)
        return self.find_fiedler_pair(L)[0]

    def grad_from_fiedler(self, fiedler_vec):
        # start = timer()
        # grad = np.array([laplacian_e.dot(fiedler_vec).dot(fiedler_vec) for laplacian_e in self.laplacian_e_list])
        grad = np.zeros(len(self.kappas))

        # grad = np.array([self.kappas[k]*(fiedler_vec[self.edge_list[k,0]]**2 +
        #                                  fiedler_vec[self.edge_list[k,1]]**2 - 2 *
        #                                  fiedler_vec[self.edge_list[k,0]]*fiedler_vec[self.edge_list[k,1]]) for
        #                  k in range(len(self.kappas))])

        for k in range(len(self.kappas)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            kappa_k = self.kappas[k]
            kdelta = kappa_k * (v_i - v_j)
            grad[k] = kdelta * (v_i - v_j)
            # grad[k] = kappa_k * (v_i**2 + v_j**2 - 2.0*v_i*v_j)

        # grad = np.array([self.kappas[]])
        # grad = grad_from_fiedler_numba(fiedler_vec, self.laplacian_e_list)
        # end = timer()
        # print("grad time: ", end - start)
        return grad

    def round_solution(self, w, k):
        idx = np.argpartition(w, -k)[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    def simple_random_round(self, w, k):
        x = np.zeros(len(w))
        for i in range(len(w)):
            r = np.random.rand()
            if w[i] > r:
                x[i] = 1.0
        return x

    def round_solution_tiebreaker(self, w, k):
        truncated_w = w.round(decimals=10)
        zipped_vals = np.array([(truncated_w[i], self.kappas[i]) for i in range(len(w))], dtype=[('weight', 'float'), ('kappa', 'float')])
        # zipped_vals = np.array([w[i]*self.kappas[i] for i in range(len(w))])
        idx = np.argpartition(zipped_vals, -k, order=['weight', 'kappa'])[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    def fw_subset(self, w_init, k, max_iters=5, duality_gap_tol=1e-8, debug_plot=True, line_search=True):
        """
        f: Python function to maximize, concave in w
        w_init: Array-like, must satisfy 0 <= w_i <= 1, |w| <= k
        k: size of max allowed subset
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
            # print(f_i)
            u_i = min(u_i, f_i + grad_i @ (s_i - w_i))
            # print(u_i)

            # print(f"Duality gap at iteration {it}: {u_i - f_i}")
            # If the duality gap is sufficiently small, we are done
            if u_i - f_i < duality_gap_tol:
                print("Duality gap tolerance reached, found optimal solution")
                print("Num unique elements: ", len(np.unique(w_i.round(decimals=5))))
                print("Num total elements: ", len(w_i))
                return self.round_solution_tiebreaker(w_i, k), w_i, u_i
                # return self.simple_random_round(w_i, k), w_i

            # Step size determination - naive method. No line search
            alpha = 2.0 / (it + 2.0)
            w_i = w_i + alpha * (s_i - w_i)

        print("Reached maximum iterations")
        print("Num unique elements: ", len(np.unique(w_i.round(decimals=5))))
        print("Num total elements: ", len(w_i))
        return self.round_solution_tiebreaker(w_i, k), w_i, u_i
