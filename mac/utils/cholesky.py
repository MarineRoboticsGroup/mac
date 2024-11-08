import math
import numpy as np
from mac.utils.graphs import *
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix
from sksparse.cholmod import cholesky, Factor, analyze, cholesky_AAt, CholmodNotPositiveDefiniteError

def update_cholesky_factorization_inplace(
        chol_factorization: Factor, edge_uv: Edge, num_nodes: int, reduced: bool, subtract: bool
) -> None:
    """Update the Cholesky factorization of the Laplacian matrix in-place
    to account for the addition of a new edge.

    Args:
        chol_factorization (Factor): the Cholesky factorization of the Laplacian
        edge_uv (Edge): the edge to add
        num_nodes (int): the number of nodes in the graph (defines the size of the Laplacian)
    """
    # the update is of a from A + C C', where C is a column vector embedded in a
    # matrix of size (n, n). This is just so we can use the cholesky
    # factorization update function
    if reduced:
        update_shape = (num_nodes - 1, num_nodes - 1)
    else:
        update_shape = (num_nodes, num_nodes)
    row_idxs = []
    val_list = []
    u, v, weight = edge_uv

    # decrement the row index by 1 if we are using the reduced Laplacian
    if reduced:
        u -= 1
        v -= 1
    sqrt_weight = math.sqrt(weight)
    if u >= 0:
        row_idxs.append(u)
        val_list.append(sqrt_weight)
    if v >= 0:
        row_idxs.append(v)
        val_list.append(-sqrt_weight)

    rows = np.array(row_idxs)
    cols = np.zeros_like(rows)

    # get the vals which should be the square root of the edge weight
    vals = np.array(val_list)

    # get the update matrix
    update_mat = csc_matrix((vals, (rows, cols)), shape=update_shape)

    # update the cholesky factorization
    chol_factorization.update_inplace(update_mat, subtract=subtract)

def get_matrix_from_chol_factor_with_original_ordering(
    chol_factorization: Factor,
) -> csc_matrix:
    """Returns the original matrix from the Cholesky factorization of the matrix
    with the original ordering of the nodes. This is useful because a
    permutation matrix may be applied to matrix to improve the sparsity pattern
    of the Cholesky factorization.

    Args:
        chol_factorization (Factor): the Cholesky factorization of the matrix

    Returns:
        csc_matrix: the original matrix
    """
    L = chol_factorization.L()
    inv_perm_order = get_inverse_permutation_ordering(chol_factorization)

    # swap the rows of the lower triangular factor to get the original ordering
    L = L[inv_perm_order, :]

    # get the original matrix
    A = L @ L.T

    assert isinstance(A, csc_matrix), f"Expected mat to be a csc_matrix, got {type(A)}"
    return A

def get_inverse_permutation_ordering(chol_factorization: Factor) -> np.ndarray:
    permutation_ordering = chol_factorization.P()
    inverse_permutation_ordering = np.empty_like(permutation_ordering)
    inverse_permutation_ordering[permutation_ordering] = np.arange(
        len(permutation_ordering)
    )
    return inverse_permutation_ordering

def get_cholesky_forward_solve(chol_factorization: Factor, b: np.ndarray) -> np.ndarray:
    """Solve the linear system Lx = b, where L is the lower triangular matrix
    from the Cholesky factorization of the matrix.

    NOTE: this is the forward solve of a potentially permuted PSD matrix, the
    norms should be consistent but the actual vector computed may differ depending
    on the ordering method

    Args:
        chol_factorization (Factor): the Cholesky factorization of the matrix

    Returns:
        np.ndarray: the solution to the linear system
    """
    diags = chol_factorization.D()
    # TODO: add checks to make sure the matrix is positive definite
    diag_vals = np.sqrt(1/ diags)
    x = diag_vals * chol_factorization.solve_L(chol_factorization.apply_P(b))
    return x

class _CholeskySolver:
    """Cholesky factorization.

    To solve Ax = b:
        solver = _CholeskySolver(A)
        x = solver.solve(b)

    optional argument `tol` on solve method is ignored but included
    to match _PCGsolver API.
    """

    def __init__(self, A):
        import scipy as sp
        import scipy.sparse.linalg  # call as sp.sparse.linalg

        try:
            self._chol = cholesky(A, beta=0)
        except CholmodNotPositiveDefiniteError as e:
            self._chol = cholesky(A, beta=1e-6)

    def solve(self, B, tol=None):
        import numpy as np

        B = np.asarray(B)
        X = np.ndarray(B.shape, order="F")
        for j in range(B.shape[1]):
            X[:, j] = self._chol(B[:, j])
        return X


def tracemin_fiedler_cholesky(L, X, normalized, tol):
    """Compute the Fiedler vector of L using the TraceMIN-Fiedler algorithm.

    It is essentially the implementation from NetworkX,
    https://networkx.org/documentation/stable/_modules/networkx/linalg/algebraicconnectivity.html#fiedler_vector
    but replaces LU decomposition with Cholesky factorization

    Parameters
    ----------
    L : (num_nodes x num_nodes) Laplacian of a possibly weighted or normalized,
    but undirected graph

    X : Initial guess for a solution. Usually a matrix of random numbers.
        This function allows more than one column in X to identify more than
        one eigenvector if desired.

    normalized : bool
        Whether the normalized Laplacian matrix is used.

    tol : float
        Tolerance of relative residual in eigenvalue computation.
        Warning: There is no limit on number of iterations.

    Returns
    -------
    sigma, X : Two NumPy arrays of floats.
        The lowest eigenvalues and corresponding eigenvectors of L.
        The size of input X determines the size of these outputs.
        As this is for Fiedler vectors, the zero eigenvalue (and
        constant eigenvector) are avoided.

    """
    import numpy as np
    import scipy as sp
    import scipy.linalg  # call as sp.linalg
    import scipy.linalg.blas  # call as sp.linalg.blas
    import scipy.sparse  # call as sp.sparse

    n = X.shape[0]

    if normalized:
        # Form the normalized Laplacian matrix and determine the eigenvector of
        # its nullspace.
        e = np.sqrt(L.diagonal())
        # TODO: rm csr_array wrapper when spdiags array creation becomes available
        D = sp.sparse.csr_array(sp.sparse.spdiags(1 / e, 0, n, n, format="csr"))
        L = D @ L @ D
        e *= 1.0 / np.linalg.norm(e, 2)

    if normalized:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= (X[:, j] @ e) * e

    else:

        def project(X):
            """Make X orthogonal to the nullspace of L."""
            X = np.asarray(X)
            for j in range(X.shape[1]):
                X[:, j] -= X[:, j].sum() / n
                pass
            pass
        pass

    # Convert A to CSC to suppress SparseEfficiencyWarning.
    A = sp.sparse.csc_array(L, dtype=float, copy=True)
    # Force A to be nonsingular. Since A is the Laplacian matrix of a
    # connected graph, its rank deficiency is one, and thus one diagonal
    # element needs to modified. Changing to infinity forces a zero in the
    # corresponding element in the solution.
    i = (A.indptr[1:] - A.indptr[:-1]).argmax()
    A[i, i] = float("inf")
    solver = _CholeskySolver(A)

    # Initialize.
    Lnorm = abs(L).sum(axis=1).flatten().max()
    project(X)
    W = np.ndarray(X.shape, order="F")

    while True:
        # Orthonormalize X.
        X = np.linalg.qr(X)[0]
        # Compute iteration matrix H.
        W[:, :] = L @ X
        H = X.T @ W
        sigma, Y = sp.linalg.eigh(H, overwrite_a=True)
        # Compute the Ritz vectors.
        X = X @ Y
        # Test for convergence exploiting the fact that L * X == W * Y.
        res = sp.linalg.blas.dasum(W @ Y[:, 0] - sigma[0] * X[:, 0]) / Lnorm
        if res < tol:
            break
        # Compute X = L \ X / (X' * (L \ X)).
        # L \ X can have an arbitrary projection on the nullspace of L,
        # which will be eliminated.
        W[:, :] = solver.solve(X, tol)
        X = (sp.linalg.inv(W.T @ X) @ W.T).T  # Preserves Fortran storage order.
        project(X)

    return sigma, np.asarray(X)


def find_fiedler_pair_cholesky(L, x, normalized, tol, seed):
    q = min(4, L.shape[0] - 1)
    X = np.asarray(seed.normal(size=(q, L.shape[0]))).T
    sigma, X = tracemin_fiedler_cholesky(L, X, normalized, tol)
    return sigma[0], X[:, 0]


class CholeskyFiedlerSolver:
    """Solver for the Fiedler vector / value of a graph using the TraceMIN-Fiedler
    algorithm with Cholesky factorization.

    This class specifically allows us to update the Cholesky factor of the
    Laplacian as new edges are added, thereby avoiding the need to recompute it
    every time we want new Fiedler vectors / values.

    It is essentially the implementation from NetworkX:
    https://networkx.org/documentation/stable/_modules/networkx/linalg/algebraicconnectivity.html#fiedler_vector
    but replaces LU decomposition with incremental Cholesky factorization:
    """
    def __init__(self, L, normalized, tol, seed):
        import scipy.sparse.linalg  # call as sp.sparse.linalg
        import scipy as sp

        self.num_nodes = L.shape[0]
        self.L = L
        self.normalized = normalized
        self.tol = tol
        self.seed = seed

        # Convert A to CSC to suppress SparseEfficiencyWarning.
        A = sp.sparse.csc_array(L, dtype=float, copy=True)
        # Force A to be nonsingular. Since A is the Laplacian matrix of a
        # connected graph, its rank deficiency is one, and thus one diagonal
        # element needs to modified. Changing to infinity forces a zero in the
        # corresponding element in the solution.
        i = (A.indptr[1:] - A.indptr[:-1]).argmax()
        A[i, i] = float("inf")
        self._chol = cholesky(A, beta=0)

    def solve_linear_system(self, B, tol=None):
        B = np.asarray(B)
        X = np.ndarray(B.shape, order="F")
        for j in range(B.shape[1]):
            X[:, j] = self._chol(B[:, j])
        return X

    def add_edge(self, edge):
        update_cholesky_factorization_inplace(self._chol, edge, self.num_nodes, reduced=False, subtract=False)
        self.L += weight_graph_lap_from_edge_list([edge], self.num_nodes)

    def remove_edge(self, edge):
        update_cholesky_factorization_inplace(self._chol, edge, self.num_nodes, reduced=False, subtract=True)
        self.L -= weight_graph_lap_from_edge_list([edge], self.num_nodes)

    def find_fiedler_pair(self, X=None, normalized=False, tol=1e-8):
        q = min(4, self.L.shape[0] - 1)
        if X is None:
            X = np.asarray(self.seed.normal(size=(q, self.L.shape[0]))).T
        sigma, X = self.tracemin_fiedler(X, normalized, tol)
        return sigma[0], X[:, 0]

    def tracemin_fiedler(self, X, normalized, tol):
        import numpy as np
        import scipy as sp
        import scipy.linalg  # call as sp.linalg
        import scipy.linalg.blas  # call as sp.linalg.blas
        import scipy.sparse  # call as sp.sparse

        n = X.shape[0]

        if normalized:
            # Form the normalized Laplacian matrix and determine the eigenvector of
            # its nullspace.
            e = np.sqrt(L.diagonal())
            # TODO: rm csr_array wrapper when spdiags array creation becomes available
            D = sp.sparse.csr_array(sp.sparse.spdiags(1 / e, 0, n, n, format="csr"))
            L = D @ L @ D
            e *= 1.0 / np.linalg.norm(e, 2)

        if normalized:

            def project(X):
                """Make X orthogonal to the nullspace of L."""
                X = np.asarray(X)
                for j in range(X.shape[1]):
                    X[:, j] -= (X[:, j] @ e) * e

        else:

            def project(X):
                """Make X orthogonal to the nullspace of L."""
                X = np.asarray(X)
                for j in range(X.shape[1]):
                    X[:, j] -= X[:, j].sum() / n
                    pass
                pass
            pass

        # Convert A to CSC to suppress SparseEfficiencyWarning.
        # A = sp.sparse.csc_array(L, dtype=float, copy=True)
        # Force A to be nonsingular. Since A is the Laplacian matrix of a
        # connected graph, its rank deficiency is one, and thus one diagonal
        # element needs to modified. Changing to infinity forces a zero in the
        # corresponding element in the solution.
        # i = (A.indptr[1:] - A.indptr[:-1]).argmax()
        # A[i, i] = float("inf")
        # solver = _CholeskySolver(A)

        # Initialize.
        Lnorm = abs(self.L).sum(axis=1).flatten().max()
        project(X)
        W = np.ndarray(X.shape, order="F")

        while True:
            # Orthonormalize X.
            X = np.linalg.qr(X)[0]
            # Compute iteration matrix H.
            W[:, :] = self.L @ X
            H = X.T @ W
            sigma, Y = sp.linalg.eigh(H, overwrite_a=True)
            # Compute the Ritz vectors.
            X = X @ Y
            # Test for convergence exploiting the fact that L * X == W * Y.
            res = sp.linalg.blas.dasum(W @ Y[:, 0] - sigma[0] * X[:, 0]) / Lnorm
            if res < tol:
                break
            # Compute X = L \ X / (X' * (L \ X)).
            # L \ X can have an arbitrary projection on the nullspace of L,
            # which will be eliminated.
            W[:, :] = self.solve_linear_system(X, tol)
            X = (sp.linalg.inv(W.T @ X) @ W.T).T  # Preserves Fortran storage order.
            project(X)

        return sigma, np.asarray(X)

