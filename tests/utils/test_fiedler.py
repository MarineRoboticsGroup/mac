"""
Copyright 2023 MIT Marine Robotics Group

Tests for Fiedler utilities

Author: Kevin Doherty
"""

import unittest
import numpy as np
import networkx as nx

from mac.utils.conversions import nx_to_mac
from mac.utils.graphs import weight_graph_lap_from_edge_list

# Code under test.
from mac.utils.fiedler import *

from networkx.linalg.algebraicconnectivity import algebraic_connectivity

class TestFiedlerUtils(unittest.TestCase):
    def setUp(self):
        # Complete graph on 5 nodes
        self.complete_graph = nx.complete_graph(5)

    def test_complete_graph(self):
        # The algebraic connectivity of the complete graph K(N) on N nodes is
        # exactly equal to N.
        edge_list = nx_to_mac(self.complete_graph)
        N = self.complete_graph.number_of_nodes()
        L = weight_graph_lap_from_edge_list(edge_list, N)
        fiedler_value, fiedler_vec, _ = find_fiedler_pair(L)
        self.assertTrue(np.isclose(fiedler_value, N))

    def test_disconnected(self):
        # Test multiple connected components
        return

    def test_cache(self):
        return


if __name__ == '__main__':
    unittest.main()
