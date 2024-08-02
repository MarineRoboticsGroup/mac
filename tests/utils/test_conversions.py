"""
Copyright 2023 MIT Marine Robotics Group

Tests for Fiedler utilities

Author: Kevin Doherty
"""

import unittest
import numpy as np
import networkx as nx

# Code under test
from mac.utils.conversions import *

def edges_equal(x: List[Edge], y: List[Edge]) -> bool:
    """
    Check for equality of two edge lists in the "MAC" format.
    """
    x.sort(key=lambda e: (e.i, e.j))
    y.sort(key=lambda e: (e.i, e.j))
    return x == y

class TestConversionsUtils(unittest.TestCase):
    def setUp(self):
        self.petersen_nx = nx.petersen_graph()
        # Manually enumerate the 15 edges of the Petersen graph
        # (cf. https://networkx.org/documentation/stable/_modules/networkx/generators/small.html#petersen_graph)
        self.petersen_mac = [Edge(0, 1, weight=1),
                             Edge(0, 4, weight=1),
                             Edge(0, 5, weight=1),
                             Edge(1, 2, weight=1),
                             Edge(1, 6, weight=1),
                             Edge(2, 3, weight=1),
                             Edge(2, 7, weight=1),
                             Edge(3, 4, weight=1),
                             Edge(3, 8, weight=1),
                             Edge(4, 9, weight=1),
                             Edge(5, 7, weight=1),
                             Edge(5, 8, weight=1),
                             Edge(6, 8, weight=1),
                             Edge(6, 9, weight=1),
                             Edge(7, 9, weight=1)]

    def test_mac_to_mac_via_nx(self):
        mac_to_mac_via_nx = nx_to_mac(mac_to_nx(self.petersen_mac))
        self.assertTrue(edges_equal(mac_to_mac_via_nx, self.petersen_mac))

    def test_nx_to_mac_equals_mac(self):
        self.assertTrue(edges_equal(nx_to_mac(self.petersen_nx), self.petersen_mac))

if __name__ == '__main__':
    unittest.main()
