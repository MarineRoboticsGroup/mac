"""
Copyright 2023 MIT Marine Robotics Group

Regression tests for MAC

Author: Kevin Doherty
"""

import unittest
import numpy as np
import networkx as nx

from mac.mac import MAC
from mac.utils import select_edges, split_edges, nx_to_mac, mac_to_nx
from .utils import get_split_petersen_graph

class TestMACRegression(unittest.TestCase):
    # Test for regressions in MAC performance
    # This test is based on the following:
    # - Algebraic connectivity for standard graphs
    # -

    def test_petersen(self):
        """
        Test the Petersen graph
        """
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        for pct_candidates in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            with self.subTest(pct_candidates=pct_candidates):
                num_candidates = int(pct_candidates * len(candidate_edges))

                # Construct an initial guess
                w_init = np.zeros(len(candidate_edges))
                w_init[:num_candidates] = 1.0

                mac = MAC(fixed_edges, candidate_edges, n)
                result, unrounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=100)
                init_selected = select_edges(candidate_edges, w_init)
                selected = select_edges(candidate_edges, result)

                init_l2 = mac.evaluate_objective(w_init)
                mac_unrounded_l2 = mac.evaluate_objective(unrounded)
                mac_l2 = mac.evaluate_objective(result)

                print("Initial L2: {}".format(init_l2))
                print("MAC Unrounded L2: {}".format(mac_unrounded_l2))
                print("MAC Rounded L2: {}".format(mac_l2))

                self.assertGreater(mac_unrounded_l2, init_l2, msg="""MAC unrounded connectivity
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

    @unittest.skip("Skipping, since this test will not pass [slight differences in eigvecs can change output]")
    def test_cache(self):
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()

        for pct_candidates in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            with self.subTest(pct_candidates=pct_candidates):
                num_candidates = int(pct_candidates * len(candidate_edges))

                # Construct an initial guess
                w_init = np.zeros(len(candidate_edges))
                w_init[:num_candidates] = 1.0

                mac = MAC(fixed_edges, candidate_edges, n)
                mac_cache = MAC(fixed_edges, candidate_edges, n, use_cache=True)

                result, unrounded, upper = mac.fw_subset(w_init, num_candidates, max_iters=100)
                result_cache, unrounded_cache, upper_cache = mac_cache.fw_subset(w_init, num_candidates, max_iters=100)

                self.assertTrue(np.allclose(unrounded, unrounded_cache), msg=f"""Cached MAC unrounded
                result should be the same as non-cached result\n MAC: {unrounded} \n Cache: {unrounded_cache}""")

                self.assertTrue(np.allclose(result, result_cache), msg=f"""Cached MAC result
                should be the same as non-cached result\n MAC: {result} \n Cache: {result_cache}""")

                self.assertTrue(np.allclose(upper, upper_cache), msg=f"""Cached MAC upper
                result should be the same as non-cached result\n MAC: {upper} \n Cache: {upper_cache}""")

                mac_unrounded_l2 = mac.evaluate_objective(unrounded)
                mac_l2 = mac.evaluate_objective(result)

                mac_cache_unrounded_l2 = mac.evaluate_objective(unrounded_cache)
                mac_cache_l2 = mac.evaluate_objective(result_cache)

                print("MAC Unrounded L2: {}".format(mac_unrounded_l2))
                print("MAC Rounded L2: {}".format(mac_l2))

                print("Cache MAC Unrounded L2: {}".format(mac_cache_unrounded_l2))
                print("Cache MAC Rounded L2: {}".format(mac_cache_l2))

                pass

if __name__ == '__main__':
    unittest.main()
