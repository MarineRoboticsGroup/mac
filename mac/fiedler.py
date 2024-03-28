import numpy as np

import networkx as nx
import networkx.linalg as la

# TRACEMIN-Fiedler imports
import scipy as sp

def find_fiedler_pair(L, method='tracemin_lu', tol=1e-8):
    """
    Compute the second smallest eigenvalue of L and corresponding
    eigenvector using `method` and tolerance `tol`.

    w: An element of [0,1]^m; this is the edge selection to use
    method: Any method supported by NetworkX for computing algebraic
    connectivity. See:
    https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html

    tol: Numerical tolerance for eigenvalue computation

    returns a tuple (lambda_2(L), v_2(L)) containing the Fiedler
    value and corresponding vector.

    """
    assert method != 'lobpcg', "lobpcg is not currently supported" # LOBPCG not supported at the moment

    seed = np.random.RandomState(7)
    q = min(4, L.shape[0] - 1)
    X = np.asarray(seed.normal(size=(q, L.shape[0]))).T
    if method == 'tracemin_cholesky':
        from mac.cholesky_utils import tracemin_fiedler_cholesky
        sigma, X = tracemin_fiedler_cholesky(L=L, X=X, normalized=False, tol=tol)
    else:
        sigma, X = la.algebraicconnectivity._tracemin_fiedler(L=L, X=X, normalized=False, tol=tol, method=method)

    return (sigma[0], X[:, 0])

def find_fiedler_pair_and_eigvecs(L, X=None, method='tracemin_lu', tol=1e-8, seed=None):
    """
    Compute the second smallest eigenvalue of L and corresponding
    eigenvector using `method` and tolerance `tol`.

    w: An element of [0,1]^m; this is the edge selection to use
    method: Any method supported by NetworkX for computing algebraic
    connectivity. See:
    https://networkx.org/documentation/stable/reference/generated/networkx.linalg.algebraicconnectivity.algebraic_connectivity.html

    X: Initial guess for eigenvectors. If None, use random initial guess.

    tol: Numerical tolerance for eigenvalue computation

    returns a tuple (lambda_2(L), v_2(L)) containing the Fiedler
    value and corresponding vector.

    """
    if seed is None:
        seed = np.random.RandomState(7)
    q = min(4, L.shape[0] - 1)
    if X is None:
        random_init = np.asarray(seed.normal(size=(q, L.shape[0]))).T
        X = random_init

    # Check if X is a valid initial guess
    assert X.shape[0] == L.shape[0]
    assert X.shape[1] == q

    if method == 'tracemin_cholesky':
        from mac.cholesky_utils import tracemin_fiedler_cholesky
        sigma, X = tracemin_fiedler_cholesky(L=L, X=X, normalized=False, tol=tol)
    else:
        sigma, X = la.algebraicconnectivity._tracemin_fiedler(L=L, X=X, normalized=False, tol=tol, method=method)

    return (sigma[0], X[:, 0], X)

def grad_from_fiedler(fiedler_vec, edge_list, weights):
    """
    Compute a (super)gradient of the algebraic connectivity with respect to w
    from the Fiedler vector.

    fiedler_vec: Eigenvector of the Laplacian corresponding to the second eigenvalue evaluated at a point 'w'.
    edge_list: np.array of shape (m,2) containing the edge list of the graph.
    weights: np.array of weights for each edge in the graph.

    returns grad F(w) from equation (8) of our paper: https://arxiv.org/pdf/2203.13897.pdf.
    """
    grad = np.zeros(len(weights))

    for k in range(len(weights)):
        edge = edge_list[k] # get edge (i,j)
        v_i = fiedler_vec[edge[0]]
        v_j = fiedler_vec[edge[1]]
        weight_k = weights[k]
        kdelta = weight_k * (v_i - v_j)
        grad[k] = kdelta * (v_i - v_j)
    return grad
