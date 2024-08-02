"""
Utilities for maximizing concave functions over simple feasible sets with Frank-Wolfe
"""

import numpy as np

def naive_stepsize(k):
    return 2.0 / (k + 2.0)

def frank_wolfe(initial,
                problem,
                solve_lp,
                stepsize=None,
                maxiter=50,
                relative_duality_gap_tol=1e-5,
                grad_norm_tol=1e-10,
                verbose=False):
    """Frank-Wolfe algorithm for maximizing a concave function.

    Parameters
    ----------
    initial : array-like
        Initial guess for the minimizer.
    problem : callable
        Function returning a tuple (f, gradf) where f is the objective function.
    solve_lp : callable
        Solve the Frank-Wolfe LP subproblem over the feasible set.
    stepsize : callable, optional
        Step size function.
    maxiter : int, optional
        Maximum number of iterations to perform.
    relative_duality_gap_tol : float, optional
        Tolerance for the *relative* duality gap (i.e., the duality gap divided
        by the objective value). If the relative duality gap is less than this
        tolerance, the algorithm terminates.
    grad_norm_tol : float, optional
        Tolerance for the norm of the gradient. If the norm of the gradient is
        less than this tolerance, the algorithm terminates.
    verbose : bool, optional
        Whether to print the iteration number and the objective value.

    Returns
    -------
    x : array-like
        The minimizer.

    """
    if stepsize is None:
        stepsize = lambda x, g, s, k: naive_stepsize(k)

    x = initial
    u = float("inf")
    for i in range(maxiter):
        # Compute objective value and a (super)-gradient.
        f, gradf = problem(x)

        # Solve the direction-finding subproblem by maximizing the linear
        # approximation of f at x over the feasible set.
        s = solve_lp(gradf)

        # Compute dual upper bound from linear approximation.
        u = min(u, f + gradf @ (s - x))

        # If the gradient norm is sufficiently small, we are done.
        if np.linalg.norm(gradf) < grad_norm_tol:
            if verbose:
                print("Gradient norm is approximately 0. Found optimal solution")
            return x, u

        # If the *relative* duality gap is sufficiently small, we are done.
        if (u - f) / abs(f) < relative_duality_gap_tol:
            if verbose:
                print("Duality gap tolerance reached, found optimal solution")
            return x, u

        x = x + stepsize(x, gradf, s, i) * (s - x)
        pass
    if verbose:
        print("Reached maximum number of iterations, returning best solution")
    return x, u

