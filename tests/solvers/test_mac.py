"""
Copyright 2023 MIT Marine Robotics Group

Regression tests for MAC

Author: Kevin Doherty
"""

import unittest
import numpy as np
import networkx as nx

from mac.utils.conversions import nx_to_mac
from mac.utils.graphs import select_edges

# Code under test
from mac.solvers.mac import MAC

class TestPetersenGraphConnected(unittest.TestCase):
    """
    Test sparsification on the Petersen graph with a connected "fixed" base graph.
    """
    def setUp(self):
        graph = nx.petersen_graph()
        # Split the graph into a tree part and a "loop" part
        spanning_tree = nx.minimum_spanning_tree(graph)
        loop_graph = nx.difference(graph, spanning_tree)
        self.fixed_edges = nx_to_mac(spanning_tree)
        self.candidate_edges = nx_to_mac(loop_graph)
        self.n = graph.number_of_nodes()

    # Test for regressions in MAC performance
    # This test is based on the following:
    # - Algebraic connectivity for standard graphs
    def test_petersen(self):
        """
        Test the Petersen graph
        """
        for pct_candidates in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            with self.subTest(pct_candidates=pct_candidates):
                num_candidates = int(pct_candidates * len(self.candidate_edges))

                # Construct an initial guess
                x_init = np.zeros(len(self.candidate_edges))
                x_init[:num_candidates] = 1.0

                mac = MAC(self.fixed_edges, self.candidate_edges, self.n)
                result, unrounded, upper = mac.solve(num_candidates, x_init, max_iters=100)
                init_selected = select_edges(self.candidate_edges, x_init)
                selected = select_edges(self.candidate_edges, result)

                init_l2 = mac.evaluate_objective(x_init)
                mac_unrounded_l2 = mac.evaluate_objective(unrounded)
                mac_l2 = mac.evaluate_objective(result)

                print("Initial L2: {}".format(init_l2))
                print("MAC Unrounded L2: {}".format(mac_unrounded_l2))
                print("MAC Rounded L2: {}".format(mac_l2))

                self.assertGreaterEqual(mac_unrounded_l2, init_l2, msg="""MAC unrounded connectivity
                should be greater than initial guess""")

                # NOTE: Stubbed out test. This assertion is not necessarily
                # (mathematically) going to be true. In order to guarantee
                # this, we would need to add a "fallback" check in the MAC
                # solver that ensures that rounding does not result in a
                # solution worse than the initial In that case, we would simply
                # select the initial guess.

                # self.assertGreater(mac_l2, init_l2, msg="""MAC rounded
                # connectivity should be greater than initial guess""")
                pass

if __name__ == '__main__':
    unittest.main()
