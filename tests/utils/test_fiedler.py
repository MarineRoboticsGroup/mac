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

class TestConnectedGraphs(unittest.TestCase):
    def setUp(self):
        # Complete graph on 5 nodes
        self.complete_graph = nx.complete_graph(5)

    def test_fiedler(self):
        # The algebraic connectivity of the complete graph K(N) on N nodes is
        # exactly equal to N.
        edge_list = nx_to_mac(self.complete_graph)
        N = self.complete_graph.number_of_nodes()
        L = weight_graph_lap_from_edge_list(edge_list, N)
        fiedler_value, fiedler_vec, _ = find_fiedler_pair(L)
        self.assertTrue(np.isclose(fiedler_value, N))

class TestDisconnectedGraphs(unittest.TestCase):
    def setUp(self):
        # Construct a disconnected graph with two connected components
        component_size = 3
        self.disconnected_graph = nx.complete_graph(component_size)
        self.disconnected_graph.add_edges_from(
            (u,v) for u in range(component_size, 2*component_size) for v in range(u+1, 2*component_size))

    @unittest.skip("Feature not yet supported.")
    def test_fiedler(self):
        # Test multiple connected components
        edge_list = nx_to_mac(self.disconnected_graph)
        N = self.disconnected_graph.number_of_nodes()
        L = weight_graph_lap_from_edge_list(edge_list, N)
        fiedler_value, fiedler_vec, _ = find_fiedler_pair(L)
        self.assertTrue(np.isclose(fiedler_value, 0))

if __name__ == '__main__':
    unittest.main()
