"""
Copyright 2023 MIT Marine Robotics Group

Regression tests for Cholesky utilities

Author: Kevin Doherty
"""
import unittest
import numpy as np

np.set_printoptions(precision=4)
from mac.utils import (
    set_incidence_vector_for_edge_inplace,
    weight_reduced_graph_lap_from_edge_list,
    weight_graph_lap_from_edge_list
    )

from mac.utils.fiedler import find_fiedler_pair
from scipy.sparse import spmatrix
from sksparse.cholmod import cholesky, Factor, analyze, cholesky_AAt
from .utils import get_split_petersen_graph, get_split_erdos_renyi_graph

# Code under test
from mac.utils.cholesky import *


ORDERING_METHODS = ["natural", "best", "amd", "metis", "nesdis", "colamd", "default"]

class TestCholeskyUtils(unittest.TestCase):
    def test_set_auv_inplace(self):
        # Get the Erdos-Renyi graph
        fixed_edges, candidate_edges, n = get_split_erdos_renyi_graph()

        auv_inplace_edge = np.zeros(n - 1)
        auv_inplace_tuple = np.zeros(n - 1)
        # iterate over all possible edges
        for e in candidate_edges:
            # set the AUV to be the edge
            auv_on_demand = np.zeros_like(auv_inplace_edge)
            set_incidence_vector_for_edge_inplace(auv_inplace_edge, e, n)
            i, j, _ = e
            edge_indices = (i, j)
            set_incidence_vector_for_edge_inplace(auv_inplace_tuple, edge_indices, n)

            i -= 1
            j -= 1
            if i >= 0:
                auv_on_demand[i] = 1
            if j >= 0:
                auv_on_demand[j] = -1

            # check that the two are the same
            self.assertTrue(np.allclose(auv_inplace_edge, auv_on_demand))
            self.assertTrue(np.allclose(auv_inplace_tuple, auv_on_demand))

    def test_update_cholesky_inplace_and_get_factored_matrix(self):
        # TODO right now we're testing that these 2 functions agree, but we
        # should try to make these tests independent of each other

        fixed_edges, candidate_edges, num_nodes = get_split_erdos_renyi_graph()
        for ord_method in ORDERING_METHODS:
            with self.subTest(ord_method=ord_method):
                laplacian = weight_reduced_graph_lap_from_edge_list(fixed_edges, num_nodes)
                laplacian_factorization = cholesky(
                    laplacian, beta=0, ordering_method=ord_method
                )
                laplacian = laplacian.toarray()

                auv = np.zeros(num_nodes - 1)
                for e in candidate_edges:
                    # set the AUV to be the edge
                    set_incidence_vector_for_edge_inplace(auv, e, num_nodes)
                    lap_update = np.outer(auv, auv) * e.weight
                    laplacian += lap_update
                    assert isinstance(laplacian, np.ndarray)

                    # update the factorization
                    update_cholesky_factorization_inplace(
                        laplacian_factorization, e, num_nodes, reduced=True, subtract=False
                    )
                    sksparse_matrix = get_matrix_from_chol_factor_with_original_ordering(
                        laplacian_factorization
                    )
                    if isinstance(sksparse_matrix, spmatrix):
                        sksparse_matrix = sksparse_matrix.toarray()

                    # check that the factorization is still valid
                    self.assertTrue(
                        np.allclose(sksparse_matrix, laplacian, atol=1e-6),
                        msg=f"""The factorization is not valid after updating with edge
                        {e}\n
                            The sksparse matrix is: \n {sksparse_matrix}\n
                            The laplacian is: \n {laplacian}\n
                            The difference is: \n {np.round(sksparse_matrix - laplacian,2)}""",
                    )

    def test_lower_triangular_solve_norm(self):
        """Test that the lower triangular solve returns a vector with the
        correct norm

        We are solving the system Lx = b, where L is a lower triangular matrix
        corresponding to the Cholesky factorization of a matrix and b is a
        vector.

        We will test that the solution is correct by comparing it to the
        solution from the numpy solve function.
        """

        def verify_solutions(np_laplacian, chol_factor, b):
            np_lower = np.linalg.cholesky(np_laplacian)
            np_solve = np.linalg.lstsq(np_lower, b, rcond=None)[0]
            sksparse_solve = get_cholesky_forward_solve(chol_factor, b)

            # verify that the norms are the same
            self.assertTrue(
                np.allclose(np.linalg.norm(np_solve), np.linalg.norm(sksparse_solve))
            )

            # double check the norm by using the inv function
            chol_factor_inv = chol_factor.inv().toarray()
            reff = b.T @ chol_factor_inv @ b
            self.assertAlmostEqual(reff, np.linalg.norm(sksparse_solve) ** 2)


        fixed_edges, candidate_edges, num_nodes = get_split_erdos_renyi_graph()

        for ord_method in ORDERING_METHODS:
            with self.subTest(ord_method=ord_method):
                laplacian = weight_reduced_graph_lap_from_edge_list(fixed_edges, num_nodes)
                laplacian_factorization = cholesky(
                    laplacian, beta=0, ordering_method=ord_method
                )
                laplacian = laplacian.toarray()

                b = np.random.rand(num_nodes - 1)
                idx = -1
                verify_solutions(laplacian, laplacian_factorization, b)

                auv = np.zeros(num_nodes - 1)
                for idx, e in enumerate(candidate_edges):
                    # get a random vector (b)
                    b = np.random.rand(num_nodes - 1)

                    # set the AUV to be the edge
                    set_incidence_vector_for_edge_inplace(auv, e, num_nodes)
                    lap_update = np.outer(auv, auv) * e.weight
                    laplacian += lap_update

                    # update the factorization
                    update_cholesky_factorization_inplace(
                        laplacian_factorization, e, num_nodes, reduced=True, subtract=False
                    )

                    # verify that the solutions are the same
                    verify_solutions(laplacian, laplacian_factorization, b)
                    pass
                pass
            pass
        pass

    def test_cholesky_fiedler(self):
        """Test that the Fiedler vector and value computed via Cholesky is correct. To
        do this we will compare to the Fiedler vector and value computed
        using LU decomposition.
        """
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()
        L = weight_graph_lap_from_edge_list(fixed_edges, n)
        l2, fied = find_fiedler_pair(L)
        l2_test, fied_test = find_fiedler_pair_cholesky(L, x=None, normalized=False, tol=1e-8, seed=np.random.RandomState(7))
        # Check that the Fiedler vector and value are the same for the
        # fixed graph.
        self.assertAlmostEqual(l2, l2_test)
        self.assertTrue(np.allclose(fied, fied_test))
        for e in candidate_edges:
            with self.subTest(e=e):
                L += e.weight * weight_graph_lap_from_edge_list([e], n)
                l2, fied = find_fiedler_pair(L)
                l2_test, fied_test = find_fiedler_pair_cholesky(L, x=None, normalized=False, tol=1e-8, seed=np.random.RandomState(7))
                # Check that the Fiedler vector and value are the same for the
                # new graph.
                self.assertAlmostEqual(l2, l2_test)
                self.assertTrue(np.allclose(fied, fied_test))
                pass
            pass
        pass

    def test_incremental_cholesky_fiedler(self):
        """Test that the Fiedler vector and value computed via Cholesky is correct after incremental updates.

        To do this we will compare to the Fiedler
        vector and value computed using LU decomposition.

        """
        # Get the Petersen graph
        fixed_edges, candidate_edges, n = get_split_petersen_graph()
        L = weight_graph_lap_from_edge_list(fixed_edges, n)
        l2, fied = find_fiedler_pair(L)

        # Make a Cholesky Eig Solver
        chol_fiedler = CholeskyFiedlerSolver(L, normalized=False, tol=1e-8, seed=np.random.RandomState(7))
        l2_test, fied_test = chol_fiedler.find_fiedler_pair()
        # Check that the Fiedler vector and value are the same for the
        # fixed graph.
        self.assertAlmostEqual(l2, l2_test)
        self.assertTrue(np.allclose(fied, fied_test))
        for e in candidate_edges[0:1]:
            with self.subTest(e=e):
                L += e.weight * weight_graph_lap_from_edge_list([e], n)
                l2, fied = find_fiedler_pair(L)

                chol_fiedler.add_edge(e)
                l2_test, fied_test = chol_fiedler.find_fiedler_pair()
                # Check that the Fiedler vector and value are the same for the
                # new graph.
                self.assertAlmostEqual(l2, l2_test)
                self.assertTrue(np.allclose(fied, fied_test))
                pass
            pass
        pass


if __name__ == "__main__":
    unittest.main()
