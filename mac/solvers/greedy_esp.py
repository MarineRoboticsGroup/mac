"""Implementation of a greedy algorithm for the k-edge selection problem from
https://arxiv.org/pdf/1604.01116.pdf. This implementation follows Algorithm 1

Problem (k-ESP+): for some M+ \subset E(Kn) \ Einit
    maximize E \subset M+  tw(G(Einit \cup E))
    subject to |E| = k

where
    tw(G) = sum_{T \in T_G} V_w(T) is the weighted number of spanning trees
    V_w(T) = prod_{e \in T} w(e) is the value of a single spanning tree T
    T_G is the set of spanning trees of G

Useful notation (roughly alphabetical):
    E: Set of edges (likely the edges being selected)
    Einit: initial set of edges taken (should form a spanning tree)
    G: Graph
    Ginit: initial graph (should be a spanning tree)
    L: Laplacian matrix
    L(w): Laplacian matrix with edge weights w
    init_laplacian: Initial Laplacian matrix
    M: Set of all edges
    all_candidate_edges: Subset of edges which are left to select (E - Einit)

"""
from sksparse.cholmod import cholesky, Factor, analyze, cholesky_AAt, CholmodNotPositiveDefiniteError
import numpy as np
from timeit import default_timer as timer

# Heap utils for maintaining a priority queue in lazy ESP
import itertools
from heapq import heappush, heappop, heapify

np.set_printoptions(precision=3, suppress=True)
from scipy.sparse import csc_matrix
from typing import List, Set, Tuple, Union, Optional
import math
from mac.utils import Edge, set_incidence_vector_for_edge_inplace
from mac.cholesky_utils import (
    update_cholesky_factorization_inplace,
    get_cholesky_forward_solve,
    weight_reduced_graph_lap_from_edge_list,
)

def compute_weighted_effective_resistances(
    xuv_arr: np.ndarray, xuv_edge_weights: np.ndarray
) -> np.ndarray:
    """Compute the effective resistance of each xuv for the given xuv_arr. The
    xuv items are stored in the rows of xuv_arr.

    Args:
        xuv_arr (np.ndarray): the xuv values
        edge_weights (np.ndarray): the edge weights

    Returns:
        np.ndarray: the effective resistances
    """
    # compute the l2-norm squared of each row
    num_xuv, num_nodes = xuv_arr.shape
    xuv_norms = np.linalg.norm(xuv_arr, axis=1) ** 2

    # multiply each row by the edge weights
    weighted_xuv_arr = xuv_norms * xuv_edge_weights

    assert weighted_xuv_arr.shape == (num_xuv,)
    return weighted_xuv_arr


def find_idx_with_max_weighted_effective_resistance(
    xuv_arr: np.ndarray, xuv_edge_weights: np.ndarray
) -> int:
    """Find the index of the xuv vector with the maximum weighted effective
    resistance.

    Args:
        xuv_arr (np.ndarray): the xuv vectors
        edge_weights (np.ndarray): the edge weights

    Returns:
        int: the index of the xuv vector with the maximum weighted effective
            resistance
    """
    num_xuv, num_nodes = xuv_arr.shape
    chunk_size = 10000

    # if num_xuv is greater than chunk_size, then we break it up into chunks of chunk_size and
    # find the max weighted effective resistance for each chunk
    num_chunks = math.ceil(num_xuv / chunk_size)
    max_weighted_resistance = -1
    max_xuv_arr_idx = -1

    # compute the weighted effective resistances for each chunk and store the max weighted effective resistance
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_xuv)
        chunk_weighted_effective_resistances = compute_weighted_effective_resistances(
            xuv_arr[start_idx:end_idx, :], xuv_edge_weights[start_idx:end_idx]
        )
        curr_chunk_max_eff_resistance_idx = np.argmax(
            chunk_weighted_effective_resistances
        )
        curr_chunk_max_eff_resistance = chunk_weighted_effective_resistances[
            curr_chunk_max_eff_resistance_idx
        ]
        if curr_chunk_max_eff_resistance > max_weighted_resistance:
            max_weighted_resistance = curr_chunk_max_eff_resistance
            max_xuv_arr_idx = start_idx + curr_chunk_max_eff_resistance_idx

    return max_xuv_arr_idx


class GreedyESP:
    def __init__(
        self,
        fixed_edges: List[Edge],
        candidate_edges: List[Edge],
        num_nodes: int,
        lazy: bool = False):
        if num_nodes == 0:
            assert len(fixed_edges) == len(candidate_edges) == 0

        # get the reduced graph laplacian by pinning the first node (trimming
        # the first row and column)
        self.L_fixed = weight_reduced_graph_lap_from_edge_list(fixed_edges, num_nodes)

        try:
            self.L_fixed_factorization = cholesky(
                self.L_fixed, beta=0
            )
        except CholmodNotPositiveDefiniteError as e:
            # check if there are any rows of all zeros
            for i in range(self.L_fixed.shape[0]):
                row_has_nonzero = np.any(self.L_fixed[i, :]).toarray().any()
                if not row_has_nonzero:
                    print(f"Row {i} of L_fixed is all zeros")
                    raise e
            self.L_fixed_factorization = cholesky(
                self.L_fixed, beta=1e-4
            )

        self.fixed_edges = fixed_edges
        self.all_candidate_edges = candidate_edges
        self.num_nodes = num_nodes
        self.edge_weights = np.array([edge.weight for edge in candidate_edges])
        self.lazy = lazy

        print(
            f"Initialized GreedyESP with {len(candidate_edges)} candidate edges for a graph with {num_nodes} nodes"
        )

    def subset(self, k: int) -> Tuple[np.ndarray, List[Edge]]:
        """
        Performs edge selection.

        Args:
            k: Number of edges to select

        Returns:
            np.ndarray: The selected edges as a boolean array
            List[Edge]: The selected edges
        """
        if self.lazy:
            return self.subset_lazy(k)
        assert k > 0
        assert len(self.all_candidate_edges) >= k
        result = np.zeros(len(self.all_candidate_edges))
        curr_chol_factorization = self.L_fixed_factorization.copy()

        selected_edges: List[Edge] = []
        candidate_edges_idxs = set(range(len(self.all_candidate_edges)))

        while len(selected_edges) < k:
            e_star, e_star_idx = self.get_best_edge(
                curr_chol_factorization, candidate_edges_idxs
            )
            result[e_star_idx] = 1.0
            selected_edges.append(e_star)
            candidate_edges_idxs.remove(e_star_idx)
            update_cholesky_factorization_inplace(
                curr_chol_factorization, e_star, self.num_nodes, reduced=True, subtract=False
            )

        return result, selected_edges

    def subsets_lazy(self, ks: List[int], verbose=False) -> Tuple[List[np.ndarray], List[Edge], Optional[List[float]]]:
        """The lazy version of subset. This version is mathematically equivalent to
        the standard version, but is often orders of magnitude faster.

        The lazy version of subset works by maintaining a priority queue of the
        candidate edges, sorted by the effective resistance of the edge. The
        basic idea is that since the objective function is submodular, we know
        that the marginal gain of adding an edge 'e' only decreases when the
        set of selected edges grows. Therefore, when we recompute the effective
        resistance for the first edge in the queue and add it back to the
        priority queue, if it remains at the front, it must be better than any
        other edges in the queue. This means that we can skip computing the
        effective resistance for all the other candidates.

        Args:
            ks (List[int]): the number of edges to select for each subset, must
            monotonically increase
            verbose (bool, optional): whether to print the progress. Defaults to
            False.

        Returns:
            Tuple[List[np.ndarray], List[Edge]]: the selected edges as a boolean

        """
        # Start the timer
        start = timer()
        assert all(ks[i] <= ks[i+1] for i in range(len(ks) - 1)), "budgets must be monotonically increasing"
        assert len(self.all_candidate_edges) >= ks[-1], "Not enough candidate edges to satisfy the largest budget"
        assert ks[0] > 0, "budgets must be positive"
        result = np.zeros(len(self.all_candidate_edges))
        results: List[np.ndarray] = []
        times: List[float] = []
        curr_chol_factorization = self.L_fixed_factorization.copy()

        selected_edges: List[Edge] = []
        candidate_edges_idxs = set(range(len(self.all_candidate_edges)))
        xuv, _ = self.get_all_xuv(curr_chol_factorization, candidate_edges_idxs)
        weighted_reffs = compute_weighted_effective_resistances(xuv, self.edge_weights)

        # Construct a priority queue of the candidate edges, sorted by the
        # effective resistance of the edge
        counter = itertools.count()
        pq = []
        for item, weight in zip(candidate_edges_idxs, weighted_reffs):
            # Priority queue is a min heap, so we negate the effective resistance
            pq.append([-weight, next(counter), item])
        heapify(pq)

        for k in ks:
            if verbose:
                print(f"Running Lazy GreedyESP for budget={k}")
            while len(selected_edges) < k:
                best_reff = float("-inf")
                best_idx = None
                while True:
                    if len(pq) == 0:
                        return
                    prev_reff, _, idx = heappop(pq)
                    prev_reff = -prev_reff
                    if best_idx == idx:
                        break

                    xuv, _ = self.get_all_xuv(curr_chol_factorization, [idx])
                    weighted_reff = compute_weighted_effective_resistances(xuv, self.edge_weights[idx])
                    # we apply negative sign due to priority queue ordering strategy
                    heappush(pq, [-weighted_reff, next(counter), idx])
                    if weighted_reff > best_reff:
                        best_reff = weighted_reff
                        best_idx = idx
                    elif weighted_reff == best_reff and best_reff == 0.0:
                        best_reff = weighted_reff
                        best_idx = idx
                result[best_idx] = 1.0
                e_star = self.all_candidate_edges[best_idx]
                selected_edges.append(e_star)
                # candidate_edges_idxs.remove(e_star_idx)
                update_cholesky_factorization_inplace(
                    curr_chol_factorization, e_star, self.num_nodes, reduced=True, subtract=False
                )
                pass
            # Store the time it took to compute the current result
            end = timer()
            times.append(end - start)
            results.append(np.copy(result))

        return results, selected_edges, times

    def subset_lazy(self, k: int, verbose: bool = False) -> Tuple[np.ndarray, List[Edge], float]:
        """A convenience function for subsets_lazy that only returns the result
        for a single budget. See subsets_lazy for more details.
        """
        results, selected_edges, times = self.subsets_lazy([k], return_times=return_time, verbose=verbose)
        res = results[0]
        time = times[0]
        return res, selected_edges, time

    def find_edge_idx_with_max_weighted_effective_resistance(
        self, xuv_arr: np.ndarray, xuv_edge_idxs: List[int]
    ) -> int:
        """Find the xuv vector with the maximum weighted effective resistance.

        Args:
            xuv_arr (np.ndarray): the xuv vectors

        Returns:
            int: the index of the xuv vector with the maximum weighted effective resistance
        """

        # get the index wrt the row of the xuv_arr
        max_xuv_arr_idx = find_idx_with_max_weighted_effective_resistance(
            xuv_arr, self.edge_weights[xuv_edge_idxs]
        )
        max_edge_idx = xuv_edge_idxs[max_xuv_arr_idx]
        return max_edge_idx

    def get_best_edge(self, curr_chol_factor: Factor, M_idxs: Set[int]) -> Edge:
        all_xuv, xuv_edge_idxs = self.get_all_xuv(curr_chol_factor, M_idxs)
        best_edge_idx = self.find_edge_idx_with_max_weighted_effective_resistance(
            all_xuv, xuv_edge_idxs
        )
        best_edge = self.all_candidate_edges[best_edge_idx]

        return best_edge, best_edge_idx

    def get_all_xuv(
        self, curr_chol_factor: Factor, M_idxs: Set[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the xuv for all edges in M_idxs as well as the edge index for
        each xuv. This is useful for computing the effective resistance of each
        edge"""
        num_nodes = self.num_nodes
        num_edges = len(M_idxs)
        xuv_arr = np.zeros((num_edges, num_nodes - 1))
        xuv_edge_idxs = np.zeros(num_edges, dtype=np.int32)
        auv = np.zeros(self.L_fixed.shape[1])
        for i, e_idx in enumerate(M_idxs):
            set_incidence_vector_for_edge_inplace(
                auv, self.all_candidate_edges[e_idx], num_nodes
            )
            xuv_arr[i] = get_cholesky_forward_solve(curr_chol_factor, auv)
            xuv_edge_idxs[i] = e_idx

        return xuv_arr, xuv_edge_idxs
