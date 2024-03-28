"""
Copyright 2023 MIT Marine Robotics Group

Regression tests for GreedyESP

Author: Kevin Doherty
"""

import unittest
import numpy as np
from scipy.sparse import spmatrix
import networkx as nx

from mac.greedy_esp_minimal import MinimalGreedyESP
from mac.greedy_esp import GreedyESP
from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx
from .utils import get_split_petersen_graph, get_split_erdos_renyi_graph


class TestGreedyESP(unittest.TestCase):
    # Test for regressions in MAC performance

    def test_petersen(self):
        """
        Test the Petersen graph
        """
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        # Construct and solve a MinimalGreedyESP instance. This is an
        # implementation of GreedyESP that naively computes effective
        # resistances using a dense pseudoinverse. It does not scale,
        # but is useful for our tests.
        minimal = MinimalGreedyESP(fixed_edges, candidate_edges, n, use_reduced_laplacian=True)

        # Construct and solve a GreedyESP instance. This is our fancy
        # implementation using Cholesky magic.
        esp = GreedyESP(fixed_edges, candidate_edges, n)

        percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for pct_candidates in percentages:
            with self.subTest(pct_candidates=pct_candidates):
                num_candidates = int(pct_candidates * len(candidate_edges))

                minimal_result, minimal_edges = minimal.subset(num_candidates)
                esp_result, esp_edges = esp.subset(num_candidates)

                # Result from both methods must be identical
                self.assertTrue(
                    np.allclose(minimal_result, esp_result),
                    msg=f"""GreedyESP result must match the result from MinimalGreedyESP. \n
                                Actual (GreedyESP):  {esp_result} \n
                                Expected (Minimal):  {minimal_result}""",
                )
                self.assertTrue(minimal_edges == esp_edges,
                                msg=f"""GreedyESP edges must match the edges
                                     from MinimalGreedyESP. \n
                                minimal_edges:  {minimal_edges} \n
                                esp_edges:  {esp_edges}""")

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

                    # Construct and solve a GreedyESP instance. This is our fancy
                    # implementation using Cholesky magic.
                    esp = GreedyESP(fixed_edges, candidate_edges, n)

                    # Construct and solve a MinimalGreedyESP instance. This is an
                    # implementation of GreedyESP that naively computes effective
                    # resistances using a dense pseudoinverse. It does not scale,
                    # but is useful for our tests.
                    minimal = MinimalGreedyESP(fixed_edges, candidate_edges, n, use_reduced_laplacian=True)

                    percentages = [0.1, 0.3, 0.6, 0.8]
                    for pct_candidates in percentages:
                        with self.subTest(pct_candidates=pct_candidates):
                            num_candidates = int(pct_candidates * len(candidate_edges))
                            esp_result, esp_edges = esp.subset(num_candidates)
                            minimal_result, minimal_edges = minimal.subset(num_candidates)

                            # Result from both methods must be identical
                            self.assertTrue(
                                np.allclose(esp_result, minimal_result),
                                msg=f"""GreedyESP result must match the result from MinimalGreedyESP. \n
                                            Actual (GreedyESP):  {esp_result} \n
                                            Expected (Minimal):  {minimal_result}""",
                            )

                            self.assertTrue(esp_edges == minimal_edges,
                                            msg=f"""GreedyESP edges must match
                                            the edges from MinimalGreedyESP. \n
                                                minimal_edges:  {minimal_edges} \n
                                                esp_edges:  {esp_edges}""")


    def test_greedy_esp_is_stateless(self):
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()
        esp = GreedyESP(fixed_edges, candidate_edges, n)

        # ensure that after solving problems none of the class variables are
        # changed

        # copy the class variables
        original_class_variables = esp.__dict__.copy()
        pct_candidates = 0.4
        num_candidates = int(pct_candidates * len(candidate_edges))
        esp.subset(num_candidates)

        # ensure that the values of the class variables are unchanged
        for key, value in esp.__dict__.items():
            with self.subTest(key=key, value=value):
                orig_value = original_class_variables[key]

                # ensure that the values are the same type
                self.assertTrue(
                    type(value) == type(orig_value),
                    msg=f"""Class variable {key} has changed type. \n
                    Actual:  {type(value)} \n
                    Expected:  {type(orig_value)}""",
                )

                if isinstance(value, np.ndarray):
                    self.assertTrue(np.allclose(value, orig_value))
                elif isinstance(value, spmatrix):
                    self.assertTrue(isinstance(orig_value, spmatrix))
                    self.assertTrue(np.allclose(value.toarray(), orig_value.toarray()))
                else:
                    self.assertEqual(value, orig_value)


if __name__ == "__main__":
    unittest.main()
