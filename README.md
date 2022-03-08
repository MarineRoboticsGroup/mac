# mac
Maximizing algebraic connectivity for pose-graph sparsification

MAC is an algorithm for solving the *maximum algebraic connectivity augmentation* problem. Specfifically, given a graph containing a (potentially empty) set of "fixed" edges and a set of "candidate" edges, as well as a cardinality constraint K, MAC tries to find the set of K cadidate edges whose addition to the fixed edges produces a graph with the largest possible [algebraic connectivity](https://en.wikipedia.org/wiki/Algebraic_connectivity). MAC does this by solving a convex relaxation of the maximum algebraic connectivity augmentation problem (which is itself NP-hard). The relaxation allows for "soft" inclusion of edges in the candidate set. When a solution to the relaxation is obtained, we _round_ it to the feasible set for the original problem.

## Getting started

