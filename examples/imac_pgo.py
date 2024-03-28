"""
Class for performing incremental PGO with iMAC
"""

import sys
import numpy as np
from mac.imac import iMAC
from mac.utils import select_edges, Edge

# SE-Sync setup
sesync_lib_path = "/Users/kevin/repos/SESync/C++/build/lib"
sys.path.insert(0, sesync_lib_path)

import PySESync

class iPGO:
    def __init__(self, budget, rounding="nearest", d=3, use_cache=False):
        """Create an empty iPGO instance.

        Parameters
        ----------
        budget: int or callable
            The budget for the iMAC algorithm. If an integer, iPGO solves with
            fixed budget. If a callable, iPGO solves with a variable budget
            determined by budget(num_nodes).
        rounding: str
            The rounding strategy to use. Options are "nearest" and "random".
        d: int
            The dimension of the PGO problem. Options are 2 and 3.
        """
        if rounding not in ["nearest", "random"]:
            raise ValueError("Rounding must be 'nearest' or 'random'")
        self.imac = iMAC(use_cache=use_cache)
        self.use_cache = use_cache
        # TODO Broken
        # if not callable(budget):
        # Fixed budget
        # budget = (lambda n: int(budget))
        self.budget = budget
        self.rounding = rounding
        self.num_poses = 0
        self.d = d # Set the problem dimension

        self.fixed_meas = []
        self.candidate_meas = []

    def is_odom(self, edge):
        # Odom edges must have e.j = e.i + 1
        if abs(edge.j - edge.i) == 1:
            return True
        return False

    def is_lc(self, edge):
        # Loop closures are only allowed to go "backward"
        # i.e. j < i
        if abs(edge.j - edge.i) > 0:
            return True
        return False

    def add_edge(self, edge):
        if self.is_odom(edge):
            self.add_odom_edge(edge)
        elif self.is_lc(edge):
            self.add_lc_edge(edge)
        else:
            raise ValueError("Edge type not recognized", edge)
        return

    def add_odom_measurements(self, measurements):
        self.fixed_meas += measurements
        # Add odom edges as fixed edges to iMAC
        edges = [Edge(meas.i, meas.j, meas.kappa) for meas in measurements]
        self.imac.add_fixed_edges(edges)
        # Update the number of poses to reflect the updated nodes in iMAC
        self.num_poses = self.imac.num_nodes

        # opts = PySESync.SESyncOpts()
        # opts.verbose=False
        # opts.r0 = self.d + 1  # Start at level d + 1 of the Riemannian Staircase
        # meas = self.fixed_meas + self.candidate_meas
        # self.sesync_result = PySESync.SESync(self.to_sesync_format(meas), opts)
        return

    def lc_order(self, edge):
        """
        Ensure that any edge added to iMAC is in the correct order for a loop
        closure, i.e. i > j
        """
        if edge.i < edge.j:
            edge = Edge(edge.j, edge.i, edge.weight)
        return edge

    def sparsify(self, commit=True):
        # DO sparsification
        budget = self.budget(self.num_poses)
        # If the budget is more than the total candidates, set budget =
        # candidates
        num_candidates = len(self.candidate_meas)
        budget = min(budget, num_candidates)
        w = np.zeros(len(self.candidate_meas))
        w[:budget] = 1.0
        w, unrounded, upper_bound = self.imac.fw_subset(w, k=self.budget(self.num_poses), max_iters=20, rounding=self.rounding)
        if commit:
            self.imac.commit(w)
            # recover sparse subset of measurements
            self.candidate_meas = select_edges(self.candidate_meas, w)
            return w, unrounded, upper_bound, self.candidate_meas
        else:
            # Return the selected edges, but don't commit them
            return w, unrounded, upper_bound, select_edges(self.candidate_meas, w)

    def add_lc_measurements(self, measurements, sparsify=True):
        self.candidate_meas += measurements
        edges = [self.lc_order(Edge(meas.i, meas.j, meas.kappa)) for meas in
                 measurements]
        self.imac.add_candidate_edges(edges)
        if sparsify:
            self.sparsify()
        # selected_meas = select_edges(self.candidate_meas, w)
        # meas = self.fixed_meas + self.candidate_meas
        # solve()
        # opts = PySESync.SESyncOpts()
        # opts.verbose=False
        # opts.r0 = self.d + 1  # Start at level d + 1 of the Riemannian Staircase
        # self.sesync_result = PySESync.SESync(self.to_sesync_format(meas), opts)
        # return self.sesync_result
        return

    def solve(self):
        meas = self.fixed_meas + self.candidate_meas
        opts = PySESync.SESyncOpts()
        opts.verbose=False
        opts.r0 = self.d + 1  # Start at level d + 1 of the Riemannian Staircase
        self.sesync_result = PySESync.SESync(self.to_sesync_format(meas), opts)
        return self.sesync_result

    def to_sesync_format(self, measurements):
        """
        Convert a set of RelativePoseMeasurement to PySESync
        RelativePoseMeasurement. Requires PySESync import. TEMP utility
        function. Since iMAC PGO depends on PySESync, we should just use SESync
        format internally.
        """
        sesync_measurements = []
        for meas in measurements:
            sesync_meas = PySESync.RelativePoseMeasurement()
            sesync_meas.i = meas.i
            sesync_meas.j = meas.j
            sesync_meas.kappa = meas.kappa
            sesync_meas.tau = meas.tau
            sesync_meas.R = meas.R
            sesync_meas.t = meas.t
            sesync_measurements.append(sesync_meas)
        return sesync_measurements



