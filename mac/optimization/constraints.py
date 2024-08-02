from mac.utils.rounding import round_nearest
import numpy as np

def solve_subset_box_lp(g, k):
    """
    Maximize the objective function
        g^T x
    subject to
        0 <= x <= 1; ||x||_0 <= k

    The closed-form solution to this problem is given by a vector with 1's in
    the positions of the largest k elements of g.
    """
    return round_nearest(g, k)

def solve_box_lp(g):
    """
    Maximize the objective function
        g^T x
    subject to
        0 <= x <= 1

    The closed-form solution to this problem is given by a vector with 1's in
    the positions of all nonnegative elements of g.

    """
    solution = np.zeros_like(g)
    solution[g >= 0.0] = 1.0
    return solution
