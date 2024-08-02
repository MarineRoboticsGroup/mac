"""
Copyright 2023 MIT Marine Robotics Group

Tests for Fiedler utilities

Author: Kevin Doherty
"""

import unittest
import numpy as np

import mac.fiedler as fiedler
from .utils import get_split_petersen_graph

class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        return

    def test_cache(self):
        # Preallocate triplets
        rows = []
        cols = []
        data = []
        for edge in edges:
            # Diagonal elem (u,u)
            rows.append(edge.i)
            cols.append(edge.i)
            data.append(edge.weight)

            # Diagonal elem (v,v)
            rows.append(edge.j)
            cols.append(edge.j)
            data.append(edge.weight)

            # Off diagonal (u,v)
            rows.append(edge.i)
            cols.append(edge.j)
            data.append(-edge.weight)

            # Off diagonal (v,u)
            rows.append(edge.j)
            cols.append(edge.i)
            data.append(-edge.weight)

        return csr_matrix(coo_matrix((data, (rows, cols)), shape=[num_nodes, num_nodes]))
        return


if __name__ == '__main__':
    unittest.main()
