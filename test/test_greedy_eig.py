"""
Copyright 2023 MIT Marine Robotics Group

Tests for GreedyEig

Author: Kevin Doherty
"""

import unittest
import numpy as np
from scipy.sparse import spmatrix
import networkx as nx

from mac.greedy_eig_minimal import MinimalGreedyEig
from mac.greedy_eig import GreedyEig
from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx
from .utils import get_split_petersen_graph, get_split_erdos_renyi_graph


class TestGreedyEig(unittest.TestCase):
    """
    Tests for the GreedyEig class
    """

    def test_petersen(self):
        """
        Test the Petersen graph
        """
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        # Construct and solve a MinimalGreedyEig instance. This is an
        # implementation of GreedyEig that naively computes the Fiedler value
        # from scratch each time. It does not scale, but is useful for testing.
        minimal = MinimalGreedyEig(fixed_edges, candidate_edges, n)

        # Construct and solve a GreedyEig instance. This is our fancy
        # implementation using Cholesky magic.
        greedy_eig = GreedyEig(fixed_edges, candidate_edges, n)

        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for pct_candidates in percentages:
            with self.subTest(pct_candidates=pct_candidates):
                num_candidates = int(pct_candidates * len(candidate_edges))

                minimal_result, minimal_edges = minimal.subset(num_candidates)
                eig_result, eig_edges = greedy_eig.subset(num_candidates)

                # Result from both methods must be identical
                self.assertTrue(
                    np.allclose(minimal_result, eig_result),
                    msg=f"""GreedyEig result must match the result from MinimalGreedyEig. \n
                                Actual (GreedyEig):  {eig_result} \n
                                Expected (Minimal):  {minimal_result}""",
                )
                self.assertTrue(minimal_edges == eig_edges,
                                msg=f"""GreedyEig edges must match the edges
                                     from MinimalGreedyEig. \n
                                minimal_edges:  {minimal_edges} \n
                                eig_edges:  {eig_edges}""")

    @unittest.skip("This test is excessively slow")
    def test_erdos_renyi(self):
        """
        Test a few Erdos-Renyi graphs
        """
        #  TODO: maybe look into seeding so tests are deterministic
        num_nodes_list = [20]
        percent_connected_list = [0.1, 0.4]
        for num_nodes in num_nodes_list:
            for percent_connected in percent_connected_list:
                with self.subTest(num_nodes=num_nodes, percent_connected=percent_connected):
                    # Get the Erdos-Renyi graph
                    fixed_edges, candidate_edges, n = get_split_erdos_renyi_graph(
                        num_nodes, percent_connected
                    )

                    # Construct and solve a GreedyEig instance. This is our fancy
                    # implementation using Cholesky magic.
                    greedy_eig = GreedyEig(fixed_edges, candidate_edges, n)

                    # Construct and solve a MinimalGreedyESP instance. This is an
                    # implementation of GreedyESP that naively computes effective
                    # resistances using a dense pseudoinverse. It does not scale,
                    # but is useful for our tests.
                    minimal = MinimalGreedyEig(fixed_edges, candidate_edges, n)

                    percentages = [0.1, 0.3, 0.6, 0.8]
                    for pct_candidates in percentages:
                        with self.subTest(pct_candidates=pct_candidates):
                            num_candidates = int(pct_candidates * len(candidate_edges))
                            eig_result, eig_edges = greedy_eig.subset(num_candidates)
                            minimal_result, minimal_edges = minimal.subset(num_candidates)

                            # Result from both methods must be identical
                            self.assertTrue(
                                np.allclose(eig_result, minimal_result),
                                msg=f"""GreedyEig result must match the result from MinimalGreedyEig. \n
                                            Actual (GreedyEig):  {eig_result} \n
                                            Expected (Minimal):  {minimal_result}""",
                            )

                            self.assertTrue(eig_edges == minimal_edges,
                                            msg=f"""GreedyEig edges must match
                                            the edges from MinimalGreedyEig. \n
                                                minimal_edges:  {minimal_edges} \n
                                                eig_edges:  {eig_edges}""")
                            pass
                        pass
                    pass
                pass
            pass
        pass

if __name__ == "__main__":
    unittest.main()
