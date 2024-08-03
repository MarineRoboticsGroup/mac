"""
Copyright 2023 MIT Marine Robotics Group

Tests for Frank-Wolfe algorithm

Author: Kevin Doherty
"""

import unittest
import numpy as np
import networkx as nx

from mac.utils.conversions import nx_to_mac
from mac.utils.graphs import weight_graph_lap_from_edge_list

from mac.optimization.constraints import *

# Code under test.
from mac.optimization.frankwolfe import *

from networkx.linalg.algebraicconnectivity import algebraic_connectivity

class TestSimpleQuadratic(unittest.TestCase):
    def test_solve_box_constraint(self):
        """
        Solve min -x^2 s.t. x_i in [0,1]. Trivially solved by x = 0.
        """
        # Create a simple concave objective function f(x) = -x^2 with gradf(x) = -2x.
        problem = lambda x: (-np.inner(x,x), -2*x)
        # Solve
        solve_lp = solve_box_lp
        N = 10
        initial = 0.5 * np.ones(N)
        x, u = frank_wolfe(initial, problem, solve_lp)
        self.assertTrue(np.allclose(x, np.zeros(N)))

    def test_solve_subset_box_constraint(self):
        """
        Solve min -x^2 s.t. x_i in [0,1], sum_i x_i = 1
        """
        problem = lambda x: (-np.inner(x,x), -2*x)
        k = 1
        solve_lp = lambda g: solve_subset_box_lp(g, k)
        N = 2
        initial = np.random.rand(N)
        initial = (k / np.sum(initial)) * initial

        x, u = frank_wolfe(initial, problem, solve_lp)
        expected = (k / N) * np.ones(N)

        # We're willing to accept a pretty loose tolerance here.
        self.assertTrue(np.allclose(x, expected, atol=0.01))

    def test_convergence_around_zero(self):
        """
        Ensure that we handle the case where f(x) \approx 0 correctly (e.g. avoid
        division by zero).
        """
        # Create a simple concave objective function f(x) = -x^2 + 0.25 with gradf(x) = -2x.
        problem = lambda x: (-np.inner(x,x) + 0.25, -2*x)

        # Set up box constraints for the solver
        solve_lp = solve_box_lp

        # Set up an initialization with f(initial) = 0
        N = 10
        initial = np.zeros(N)
        initial[0] = 0.5

        # Solve
        x, u = frank_wolfe(initial, problem, solve_lp)

        self.assertTrue(np.allclose(x, np.zeros(N)))

if __name__ == '__main__':
    unittest.main()
