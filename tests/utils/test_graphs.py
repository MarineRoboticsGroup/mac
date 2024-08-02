"""
Copyright 2023 MIT Marine Robotics Group

Tests for graph utilities.

Author: Kevin Doherty
"""

import unittest
import networkx as nx
import numpy as np

from mac.utils.conversions import nx_to_mac

# Code under test
from mac.utils.graphs import *

class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        self.graph = nx.petersen_graph()
        self.weighted_graph = nx.petersen_graph()
        rng = np.random.default_rng(seed=7)
        weights = rng.random(self.weighted_graph.number_of_edges())
        for i, e in enumerate(self.weighted_graph.edges()):
            self.weighted_graph[e[0]][e[1]]['weight'] = weights[i]

    def test_weight_graph_lap_from_edge_list_unweighted(self):
        edge_list = nx_to_mac(self.graph)
        L_ours = weight_graph_lap_from_edge_list(edge_list, self.graph.number_of_nodes())
        L_nx = nx.laplacian_matrix(self.graph)
        self.assertTrue(np.allclose(L_ours.todense(), L_nx.todense()))

    def test_weight_graph_lap_from_edge_list_unweighted(self):
        edge_list = nx_to_mac(self.weighted_graph)
        L_ours = weight_graph_lap_from_edge_list(edge_list, self.graph.number_of_nodes())
        L_nx = nx.laplacian_matrix(self.weighted_graph)
        self.assertTrue(np.allclose(L_ours.todense(), L_nx.todense()))


if __name__ == '__main__':
    unittest.main()
