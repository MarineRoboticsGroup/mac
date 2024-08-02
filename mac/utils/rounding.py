"""
Utilities for rounding solutions to convex optimization problems onto simple constraint sets.
"""

def round_nearest(w, k, weights=None, break_ties_decimal_tol=None):
    """
    Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to a
    solution to the original problem with w_i \in {0,1}. Ties between edges
    are broken based on the original weight (all else being equal, we
    prefer edges with larger weight).

    w: A solution in the feasible set for the relaxed problem
    weights: The original weights of the edges to use as a tiebreaker
    k: The number of edges to select
    break_ties_decimal_tol: tolerance for determining floating point equality of weights w. If two selection weights are equal to this many decimal places, we break the tie based on the original weight.

    returns w': A solution in the feasible set for the original problem
    """
    if weights is None or break_ties_decimal_tol is None:
        # If there are no tiebreakers, just set the top k elements of w to 1,
        # and the rest to 0
        idx = np.argpartition(w, -k)[-k:]
        rounded = np.zeros(len(w))
        if k > 0:
            rounded[idx] = 1.0
        return rounded

    # If there are tiebreakers, we truncate the selection weights to the
    # specified number of decimal places, and then break ties based on the
    # original weights
    truncated_w = w.round(decimals=break_ties_decimal_tol)
    zipped_vals = np.array(
        [(truncated_w[i], weights[i]) for i in range(len(w))],
        dtype=[("w", "float"), ("weight", "float")],
    )
    idx = np.argpartition(zipped_vals, -k, order=["w", "weight"])[-k:]
    rounded = np.zeros(len(w))
    if k > 0:
        rounded[idx] = 1.0
    return rounded

def round_random(w, k):
    """
    Round a solution w to the relaxed problem, i.e. w \in [0,1]^m, |w| = k to
    one with hard edge constraints and satisfying the constraint that the
    expected number of selected edges is equal to k.

    w: A solution in the feasible set for the relaxed problem
    k: The number of edges to select _in expectation_

    returns w': A solution containing hard edge selections with an expected
    number of selected edges equal to k.
    """
    x = np.zeros(len(w))
    for i in range(len(w)):
        r = np.random.rand()
        if w[i] > r:
            x[i] = 1.0
    return x

def round_madow(w, k, seed=None, value_fn=None, max_iters=1):
    if value_fn is None or max_iters == 1:
        return round_madow_base(w, k, seed)

    best_x = None
    best_val = -np.inf
    for i in range(max_iters):
        x = round_madow_base(w, k, seed)
        val = value_fn(x)
        if val > best_val:
            best_val = val
            best_x = x
    return best_x


def round_madow_base(w, k, seed=None):
    """
    Use Madow rounding
    """
    if seed is None:
        u = np.random.rand()
    else:
        u = seed.rand()
    x = np.zeros(len(w))
    pi = np.zeros(len(w))
    sumw = np.cumsum(w)
    pi[1:] = sumw[:-1]
    for i in range(k):
        total = u + i
        x[np.where((pi <= total) & (total < sumw))] = 1.0

    assert np.sum(x) == k, f"Error: {np.sum(x)} != {k}"
    return x
