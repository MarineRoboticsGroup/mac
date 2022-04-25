# mac
Maximizing algebraic connectivity for pose-graph sparsification

MAC is an algorithm for solving the *maximum algebraic connectivity augmentation* problem. Specfifically, given a graph containing a (potentially empty) set of "fixed" edges and a set of "candidate" edges, as well as a cardinality constraint K, MAC tries to find the set of K cadidate edges whose addition to the fixed edges produces a graph with the largest possible [algebraic connectivity](https://en.wikipedia.org/wiki/Algebraic_connectivity). MAC does this by solving a convex relaxation of the maximum algebraic connectivity augmentation problem (which is itself NP-hard). The relaxation allows for "soft" inclusion of edges in the candidate set. When a solution to the relaxation is obtained, we _round_ it to the feasible set for the original problem.

## Getting started

First, get the dependencies (optionally, do this in a virtual environment).
```bash
pip install -r requirements.txt
```

Then install MAC locally with
```bash
pip install -e .
```

Now you are ready to use MAC.

## Running the examples

For the pose graph examples, you will need to install
[SE-Sync](https://github.com/david-m-rosen/SESync) with Python bindings.

## Reference

If you found this code useful, please cite:
```
@article{doherty2022spectral,
  title={Spectral Measurement Sparsification for Pose-Graph SLAM},
  author={Doherty, Kevin J and Rosen, David M and Leonard, John J},
  journal={arXiv preprint arXiv:2203.13897},
  year={2022}
}
```

## Notes

- Currently `mac.py` assumes that there is at most one candidate edge between
  any pair of nodes. If your data includes multiple edges between the same pair
  of nodes, you can combine the edges into a single edge with weight equal to
  the sum of the individual edge weights before using MAC.
