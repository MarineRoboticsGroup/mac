from mac.utils import *
from mac.mac import MAC
import numpy as np
import networkx as nx
import networkx.linalg as la
import networkx.generators as gen
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, lobpcg

from timeit import default_timer as timer

class iMAC(MAC):

    def __init__(self, fixed_edges=None, candidate_edges=None, num_nodes=0,
                fiedler_method='tracemin_lu', use_cache=False, fiedler_tol=1e-8,
                min_selection_weight_tol=1e-10):
        # Set up empty lists for fixed and candidate edges if they are not passed.
        # See https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments
        if fixed_edges is None:
            fixed_edges = []
        if candidate_edges is None:
            candidate_edges = []
        MAC.__init__(self, fixed_edges, candidate_edges, num_nodes,
                     fiedler_method, use_cache, fiedler_tol, min_selection_weight_tol)
        self.fixed_edges = fixed_edges
        self.fixed_weights = []
        for edge in fixed_edges:
            self.fixed_weights.append(edge.weight)
            pass

    def add_candidate_edges(self, edges: List[Edge]):
        """
        Add new candidate edges
        """

        # Convert edge_list back to a list (from np.array)
        self.edge_list = [e for e in self.edge_list]
        # Convert weights bac to a list (from np.array)
        self.weights = [w for w in self.weights]
        # Convert laplacian_e_list back to a list
        self.laplacian_e_list = [laplacian_e for laplacian_e in self.laplacian_e_list]

        for edge in edges:
            # For now, candidate edges cannot add new nodes
            assert(edge.i < self.num_nodes)
            assert(edge.j < self.num_nodes)

            laplacian_e = weight_graph_lap_from_edge_list([edge], self.num_nodes)
            self.weights.append(edge.weight)
            self.edge_list.append((edge.i, edge.j))
            self.laplacian_e_list.append(laplacian_e)
            pass

        # Convert everything back to np.array
        self.edge_list = np.array(self.edge_list)
        self.weights = np.array(self.weights)
        self.laplacian_e_list = np.array(self.laplacian_e_list)
        return


    def add_fixed_edges(self, edges: List[Edge]):
        """
        Add new fixed edges, reshaping Laplacian if needed
        """
        resize_laplacian = False
        for edge in edges:
            if edge.i >= self.num_nodes:
                self.num_nodes = edge.i + 1
                resize_laplacian = True
                pass

            if edge.j >= self.num_nodes:
                self.num_nodes = edge.j + 1
                resize_laplacian = True
                pass

            self.fixed_edges.append(edge)
            self.fixed_weights.append(edge.weight)
            pass

        if resize_laplacian:
            # New node added, recompute Laplacian
            self.L_fixed = weight_graph_lap_from_edge_list(self.fixed_edges, self.num_nodes)
        else:
            # No new nodes, simply update the Laplacian
            self.L_fixed += weight_graph_lap_from_edge_list(edges, self.num_nodes)
            pass
        return

    def commit(self, selection):
        """
        Commit the selection.

        NOTE: We currently do not check whether `selection` ensures a connected
        graph. In the future we should protect against user-specified inputs
        that result in disconnected graphs.

        """
        # Ensure selection is the correct length
        assert(len(selection) == len(self.edge_list))

        # Preserve only the subset of edges that were selected
        self.edge_list = self.edge_list[np.where(selection == 1.0)]
        self.lapacian_e_list = self.laplacian_e_list[np.where(selection == 1.0)]
        self.weights = self.weights[np.where(selection == 1.0)]

        return

if __name__ == '__main__':
    unittest.main()
