"""
Copyright 2023 MIT Marine Robotics Group

Regression tests to ensure that GreedyESP returns the same result whether or not
we are using the reduced Laplacian

Author: Alan Papalia
"""
import unittest
import numpy as np
import networkx as nx

from mac.greedy_esp_minimal import MinimalGreedyESP
from mac.greedy_esp import GreedyESP
from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx, get_incidence_vector
from .utils import get_split_petersen_graph


class TestGreedyEspMinimal(unittest.TestCase):

    def test_petersen(self):
        """
        Test the Petersen graph
        """
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        greedy_minimal_reduced = MinimalGreedyESP(
            fixed_edges, candidate_edges, n, use_reduced_laplacian=True
        )
        greedy_minimal_not_reduced = MinimalGreedyESP(
            fixed_edges, candidate_edges, n, use_reduced_laplacian=False
        )

        for pct_candidates in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            with self.subTest(pct_candidates=pct_candidates):
                num_candidates = int(pct_candidates * len(candidate_edges))

                # Construct and solve a MinimalGreedyESP instance. This is an
                # implementation of GreedyESP that naively computes effective
                # resistances using a dense pseudoinverse. It does not scale,
                # but is useful for our tests.
                minimal_reduced_result, reduced_edges_selected = greedy_minimal_reduced.subset(num_candidates)
                minimal_not_reduced_result, not_reduced_edges_selected = greedy_minimal_not_reduced.subset(
                    num_candidates
                )

                # Result from both methods must be identical
                self.assertTrue(
                    np.allclose(minimal_reduced_result, minimal_not_reduced_result),
                    msg=f"""GreedyMinimalReduced result must match the result from GreedyMinimalNotReduced. \n
                    (GreedyMinimalNotReduced): {minimal_not_reduced_result} \n
                    (GreedyMinimalReduced): {minimal_reduced_result}""",
                )
                self.assertTrue(
                    reduced_edges_selected == not_reduced_edges_selected,
                    msg=f"""GreedyMinimalReduced edges selected must match the
                         result from GreedyMinimalNotReduced. \n
                    (GreedyMinimalNotReduced): {not_reduced_edges_selected} \n
                    (GreedyMinimalReduced): {reduced_edges_selected}""",
                )

    def test_reff_computation(self):
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        greedy_minimal_reduced = MinimalGreedyESP(
            fixed_edges, candidate_edges, n, use_reduced_laplacian=True
        )
        greedy_minimal_not_reduced = MinimalGreedyESP(
            fixed_edges, candidate_edges, n, use_reduced_laplacian=False
        )

        # make sure they have the same edge list so we can iterate over just one
        # of them
        assert (greedy_minimal_reduced.edge_list == greedy_minimal_not_reduced.edge_list).all()

        # run the Reff computation for 10 random assignments of weights and all
        # of the edges
        num_trials = 10
        edge_list = greedy_minimal_reduced.edge_list
        for _ in range(num_trials):
            rand_weights = np.random.rand(len(greedy_minimal_reduced.edge_list))

            for edge in edge_list:
                auv = get_incidence_vector(edge, n)
                reff_reduced = greedy_minimal_reduced.compute_reff(rand_weights, auv)
                reff_not_reduced = greedy_minimal_not_reduced.compute_reff(rand_weights, auv)

                self.assertTrue(np.allclose(reff_reduced, reff_not_reduced),
                    msg=f"""Reff computation must be the same for both
                         methods.\n
                    Reff (Reduced): {reff_reduced} \n
                    Reff (Not Reduced): {reff_not_reduced}""")




if __name__ == "__main__":
    unittest.main()