from mac.utils import *
import mac.fiedler as fiedler
import mac.frankwolfe as fw
from timeit import default_timer as timer
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx
import networkx.linalg as la
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, csr_matrix

from timeit import default_timer as timer

class MAC:
    @dataclass
    class Cache:
        """Problem data cache"""
        Q: Optional[np.ndarray] = None

    def __init__(self, fixed_edges, candidate_edges, num_nodes,
                fiedler_method='tracemin_lu', fiedler_tol=1e-8,
                min_selection_weight_tol=1e-10):
        """Parameters
        ----------
        fixed_edges : list of Edge
            List of edges that are fixed in the graph.
        candidate_edges : list of Edge
            List of edges that are candidates for addition to the graph.
        num_nodes : int
            Number of nodes in the graph.
        fiedler_method : str, optional
            Method to use for computing the Fiedler vector. Options are
            'tracemin_lu', 'tracemin_cholesky'. Default is 'tracemin_lu'. Using the
            'tracemin_cholesky' method is faster but requires SuiteSparse to be
            installed.
        fiedler_tol : float, optional
            Tolerance for computing the Fiedler vector and corresponding eigenvalue.
        min_edge_selection_tol : float, optional
            Tolerance for the minimum edge selection weight. Default is 1e-10.
        """
        if (num_nodes == 0):
            assert(len(fixed_edges) == len(candidate_edges) == 0)
        self.L_fixed = weight_graph_lap_from_edge_list(fixed_edges, num_nodes)
        self.num_nodes = num_nodes
        self.weights = []
        self.edge_list = []

        for edge in candidate_edges:
            self.weights.append(edge.weight)
            self.edge_list.append((edge.i, edge.j))

        self.weights = np.array(self.weights)
        self.edge_list = np.array(self.edge_list)

        # Configuration for Fiedler vector computation
        self.fiedler_method = fiedler_method
        self.fiedler_tol = fiedler_tol

        # Truncate edges with selection weights below this threshold
        self.min_selection_weight_tol = min_selection_weight_tol

    def laplacian(self, x):
        """
        Construct the combined Laplacian (fixed edges plus candidate edges weighted by x).

        x: An element of [0,1]^m; this is the edge selection to use
        tol: Tolerance for edges that are numerically zero. This improves speed
        in situations where edges are not *exactly* zero, but close enough that
        they have almost no influence on the graph.

        returns the matrix L(w)
        """
        idx = np.where(x > self.min_selection_weight_tol)
        prod = x[idx]*self.weights[idx]
        L_candidate = weight_graph_lap_from_edges(self.edge_list[idx], prod, self.num_nodes)
        L_x = self.L_fixed + L_candidate
        return L_x

    def evaluate_objective(self, x):
        """
        Compute lambda_2(L(x)) where L(x) is the Laplacian with edge i weighted
        by x_i*weight_i and lambda_2 is the second smallest eigenvalue (this is the
        algebraic connectivity).

        x: Weights for each candidate edge (does not include fixed edges)

        returns F(x) = lambda_2(L(x)).
        """
        return fiedler.find_fiedler_pair(self.laplacian(x), method=self.fiedler_method, tol=self.fiedler_tol)[0]

    def problem(self, x, cache=None):
        """
        Compute the algebraic connectivity of L(x) and a (super)gradient of the algebraic connectivity with respect to x.

        returns x, grad F(x).
        """
        Q = None if cache is None else cache.Q
        fiedler_value, fiedler_vec, Qnew = fiedler.find_fiedler_pair(x, X=Q)

        gradf = np.zeros(len(self.weights))
        for k in range(len(self.weights)):
            edge = self.edge_list[k] # get edge (i,j)
            v_i = fiedler_vec[edge[0]]
            v_j = fiedler_vec[edge[1]]
            weight_k = self.weights[k]
            kdelta = weight_k * (v_i - v_j)
            gradf[k] = kdelta * (v_i - v_j)

        if cache is not None:
            cache.Q = Q
        return f, gradf, cache

    def solve(self, k, x_init=None, rounding="nearest", fallback=False,
                  max_iters=5, relative_duality_gap_tol=1e-4,
                  grad_norm_tol=1e-8, random_rounding_max_iters=1, verbose=False, return_rounding_time=False, use_cache=False):
        """Use the Frank-Wolfe method to solve the subset selection problem,.

        Parameters
        ----------
        x_init : Array-like
            Initial weights for the candidate edges, must satisfy 0 <= w_i <= 1, |w| <= k. This
            is the starting point for the Frank-Wolfe algorithm. TODO(kevin): make optional
        k : int
            Number of edges to select.
        rounding : str, optional
            Rounding method to use. Options are "nearest" (default) and "madow"
            (a random rounding procedure).
        fallback : bool, optional
            If True, fall back to the initialization if the rounded solution is worse.
        max_iters: int, optional
            Maximum number of iterations for the Frank-Wolfe algorithm.
        relative_duality_gap_tol: float, optional
            Tolerance for the relative duality gap, expressed as a fraction of
            the function value. That is, if (upper - f)/f <
            relative_duality_gap_tol, where "upper" is an upper bound on the
            optimal value of 'f', the algorithm terminates.
        grad_norm_tol: float, optional
            Tolerance for the norm of the gradient. If the norm of the gradient
            is less than this value, then the algorithm terminates.
        random_rounding_max_iters: int, optional
            Maximum number of iterations for the random rounding procedure.
            This is only used if rounding="madow". If this is larger than 1,
            then we will randomly round multiple times and return the best
            solution (in terms of algebraic connectivity).
        verbose: bool, optional
            If True, print out information about the progress of the algorithm.

        returns a tuple (solution, unrounded, upper_bound) where
        solution: the (rounded) solution w \in {0,1}^m |w| = k
        unrounded: the solution obtained prior to rounding
        upper_bound: the value of the dual at the last iteration

        """

        if k >= len(self.weights):
            # If the budget is larger than the number of candidate edges, then
            # keep them all.
            result = np.ones(len(self.weights))
            if return_rounding_time:
                return result, result, self.evaluate_objective(np.ones(len(self.weights))), 0.0

            return result, result, self.evaluate_objective(np.ones(len(self.weights)))

        assert(len(w_init) == len(self.weights))

        # Solution for the direction-finding subproblem
        solve_lp = lambda g: fw.solve_subset_box_lp(g, k)

        # handle case where x is none

        # Set up problem to use cache (or not)
        cache = None
        if use_cache:
            cache = Cache()
        problem = lambda x: self.problem(x, cache=cache)

        # Run Frank-Wolfe to solve the relaxation of subset constrained
        # algebraic connectivity maximization
        if self.use_cache:
            w, u = fw.frank_wolfe_with_recycling(initial=x_init,
                                                 problem=problem,
                                                 solve_lp=solve_lp,
                                                 maxiter=max_iters,
                                                 relative_duality_gap_tol=relative_duality_gap_tol,
                                                 grad_norm_tol=grad_norm_tol,
                                                 verbose=verbose)
        else:
            w, u = fw.frank_wolfe(initial=x_init, problem=self.problem,
                                  solve_lp=solve_lp, maxiter=max_iters,
                                  relative_duality_gap_tol=relative_duality_gap_tol,
                                  grad_norm_tol=grad_norm_tol,
                                  verbose=verbose)

        start = timer()
        if rounding == "madow":
            rounded = round_madow(w, k, value_fn=self.evaluate_objective, max_iters=random_rounding_max_iters)
        else:
            # rounding == "nearest"
            rounded = round_nearest(w, k, weights=self.weights, break_ties_decimal_tol=10)
        end = timer()
        rounding_time = end - start

        if fallback:
            init_f = self.evaluate_objective(x_init)
            rounded_f = self.evaluate_objective(rounded)

            # If the rounded solution is worse than the initial solution, then
            # return the initial solution instead.
            if rounded_f < init_f:
                rounded = w_init

        # Return the rounded solution along with the unrounded solution and
        # dual upper bound
        if return_rounding_time:
            return rounded, w, u, rounding_time

        return rounded, w, u
