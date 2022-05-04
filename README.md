# mac
Maximizing algebraic connectivity for graph sparsification

MAC is an algorithm for solving the *maximum algebraic connectivity augmentation* problem. Specfifically, given a graph containing a (potentially empty) set of "fixed" edges and a set of "candidate" edges, as well as a cardinality constraint K, MAC tries to find the set of K cadidate edges whose addition to the fixed edges produces a graph with the largest possible [algebraic connectivity](https://en.wikipedia.org/wiki/Algebraic_connectivity). MAC does this by solving a convex relaxation of the maximum algebraic connectivity augmentation problem (which is itself NP-hard). The relaxation allows for "soft" inclusion of edges in the candidate set. When a solution to the relaxation is obtained, we _round_ it to the feasible set for the original problem.

## Getting started

Install MAC locally with
```bash
pip install -e .
```

Now you are ready to use MAC.

## Running the examples

### Basic examples

From the `examples` directory, run:
```bash
python3 random_graph_sparsification.py
```
which demonstrates our sparsification approach on an [Erdos-Renyi graph](https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model).

In the same directoy, running:
```bash
python3 petersen_graph_sparsification.py
```
will show the results of our approach on the [Petersen graph](https://en.wikipedia.org/wiki/Petersen_graph).

In each case, the set of fixed edges is a chain, and the remaining edges are considered candidates.

### Pose graph sparsification

For the pose graph examples, you will need to install [SE-Sync](https://github.com/david-m-rosen/SE-Sync) with [Python bindings](https://github.com/david-m-rosen/SE-Sync#python).

Once that is installed, you need to modify the SE-Sync path in `g2o_experiment.py`:

```python
# SE-Sync setup
sesync_lib_path = "/path/to/SESync/C++/build/lib"
sys.path.insert(0, sesync_lib_path)
```

Finally, run:
```bash
python3 g2o_experiment.py [path to .g2o file]
```
to run MAC for pose graph sparsification and compute SLAM solutions. Several plots will be saved in the `examples` directory for inspection.

## Reference

If you found this code useful, please cite:
```bibtex
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
