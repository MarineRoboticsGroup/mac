"""
Copyright 2023 MIT Marine Robotics Group

Unit tests for iMAC

Author: Kevin Doherty
"""
import numpy as np

import unittest
from mac.utils import Edge
from mac.mac import MAC
from mac.imac import iMAC

from .utils import get_split_petersen_graph

class TestIMAC(unittest.TestCase):

    def test_constructor(self):
        """
        Construct iMAC instances.
        """
        # Create an *empty* iMAC instance
        imac = iMAC()
        self.assertEqual(len(imac.fixed_edges),0)
        self.assertEqual(len(imac.edge_list),0)

        # Test iMAC constructor with some data
        fixed_edges, candidate_edges, n = get_split_petersen_graph()
        imac = iMAC(fixed_edges, candidate_edges, n)
        self.assertEqual(len(imac.fixed_edges),len(fixed_edges))
        self.assertEqual(len(imac.edge_list),len(candidate_edges))

        # Ensure no side-effects in constructor
        imac = iMAC()
        self.assertEqual(len(imac.fixed_edges),0)
        self.assertEqual(len(imac.edge_list),0)

    def test_petersen_matches_batch(self):
        """Test that the unrounded and rounded iMAC solutions match the batch MAC solution for a Petersen graph.
        """
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        # Create an *empty* iMAC instance
        imac = iMAC()
        for edge in fixed_edges:
            imac.add_fixed_edges([edge])
        for edge in candidate_edges:
            imac.add_candidate_edges([edge])

        # Create a batch MAC instance
        mac = MAC(fixed_edges, candidate_edges, n)

        pct_candidates = 0.5
        num_candidates = int(pct_candidates * len(candidate_edges))

        # Construct an initial guess
        w_init = np.zeros(len(candidate_edges))
        w_init[:num_candidates] = 1.0

        imac_solution, imac_unrounded, imac_upper = imac.fw_subset(w_init, num_candidates, max_iters=50, rounding="nearest")
        mac_solution, mac_unrounded, mac_upper = mac.fw_subset(w_init, num_candidates, max_iters=50, rounding="nearest")

        self.assertTrue(np.allclose(mac_unrounded, imac_unrounded), msg=f"iMAC solution does not match MAC solution\niMAC: {imac_solution}\nMAC: {mac_solution}")
        self.assertTrue(np.allclose(mac_solution, imac_solution), msg=f"iMAC solution does not match MAC solution\niMAC: {imac_solution}\nMAC: {mac_solution}")

    def test_incremental_commit(self):
        """
        Test that the iMAC `commit` function behaves as expected
        """
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        # Create an *empty* iMAC instance
        imac = iMAC()
        for edge in fixed_edges:
            imac.add_fixed_edges([edge])
        for edge in candidate_edges:
            imac.add_candidate_edges([edge])

        # Ensure that iMAC has the correct number of edges stored to begin with
        self.assertEqual(len(imac.edge_list), len(candidate_edges))

        pct_candidates = 0.5
        num_candidates = int(pct_candidates * len(candidate_edges))

        # Construct a sparse selector
        w = np.zeros(len(candidate_edges))
        w[:num_candidates] = 1.0

        # Commit it to the iMAC instance
        imac.commit(w)

        self.assertEqual(len(imac.edge_list), num_candidates)

    def test_incremental_petersen(self):
        """
        Test full iMAC sparsification pipeline
        """
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        # Ensure fixed edges are sorted incrementally
        sorted_fixed_edges = []
        for e in fixed_edges:
            if e.j < e.i:
                # Swap indices
                e = Edge(e.j, e.i, e.weight)
            sorted_fixed_edges.append(e)
            pass

        # Ensure candidate edges sorted incrementally
        sorted_candidate_edges = []
        for e in candidate_edges:
            if e.j > e.i:
                # Swap indices
                e = Edge(e.j, e.i, e.weight)
            sorted_candidate_edges.append(e)
            pass

        pct_candidates = 0.5
        num_candidates = int(pct_candidates * len(sorted_candidate_edges))
        # Process nodes in order
        imac = iMAC()
        imac_solution = None
        for i in range(n):
            with self.subTest(i=i):
                # Add fixed edges
                fixed_e = [e for e in sorted_fixed_edges if e.i == i]
                imac.add_fixed_edges(fixed_e)

                # Add candidate edges
                cand_e = [e for e in sorted_candidate_edges if e.i == i]
                imac.add_candidate_edges(cand_e)

                # Construct a sparse selector, reusing prev. soln. if available
                if imac_solution is None:
                    w = np.zeros(len(imac.edge_list))
                    w[:min(num_candidates, len(imac.edge_list))] = 1.0
                else:
                    w = np.zeros(len(imac.edge_list))
                    w[:len(imac_solution)] = imac_solution

                # Solve the iMAC instance
                imac_solution, imac_unrounded, imac_upper = imac.fw_subset(w, num_candidates, max_iters=50)

                # Commit it to the iMAC instance
                imac.commit(imac_solution)
                # Consolidate the iMAC solution
                imac_solution = imac_solution[np.where(imac_solution == 1.0)]
                self.assertEqual(len(imac.edge_list),len(imac_solution))

    def test_empty_subset(self):
        """Test that the iMAC `fw_subset` function behaves as expected when there are
        no edges.

        """
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        # Ensure fixed edges are sorted incrementally
        sorted_fixed_edges = []
        for e in fixed_edges:
            # NOTE: This was the old way, that was wrong
            if e.j > e.i:
                # Swap indices
                e = Edge(e.j, e.i, e.weight)
            sorted_fixed_edges.append(e)
            pass

        # Ensure candidate edges sorted incrementally
        sorted_candidate_edges = []
        for e in candidate_edges:
            if e.j > e.i:
                # Swap indices
                e = Edge(e.j, e.i, e.weight)
            sorted_candidate_edges.append(e)
            pass

        pct_candidates = 0.5
        num_candidates = int(pct_candidates * len(sorted_candidate_edges))

        # Process nodes in order
        imac = iMAC()
        w = np.zeros(len(imac.edge_list))

        # Solve the iMAC instance
        imac_solution, imac_unrounded, imac_upper = imac.fw_subset(w, num_candidates, max_iters=50)

        self.assertEqual(len(imac_solution), 0)
        self.assertEqual(imac_upper, 0.0)

if __name__ == '__main__':
    unittest.main()


