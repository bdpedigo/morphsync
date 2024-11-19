import numpy as np

from .base import FacetFrame
from .integration import import_pyvista


class Graph(FacetFrame):
    def __init__(self, graph):
        if hasattr(graph, "vertices") and hasattr(graph, "edges"):
            vertices = graph.vertices
            edges = graph.edges
        elif isinstance(graph, tuple):
            vertices, edges = graph
        super().__init__(vertices, edges)
        self._graph = graph

    def __repr__(self):
        return f"Graph(nodes={self.nodes.shape}, edges={self.edges.shape})"

    @property
    def edges(self):
        return self.facets.values

    @property
    def edges_df(self):
        return self.facets

    @property
    def edges_positional(self):
        return np.vectorize(self.nodes.index.get_loc)(self.edges)

    def to_pyvista(self):
        pv = import_pyvista()
        lines = np.empty((len(self.edges_positional), 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1:3] = self.edges_positional

        return pv.PolyData(self.vertices, lines=lines)

    def to_adjacency(self, return_as="csr", weights=None, directed=False):
        if return_as == "csr":
            from scipy.sparse import csr_array

            n = self.n_nodes
            if weights is None:
                data = np.ones(len(self.edges))
            else:
                raise NotImplementedError("Weighted adjacency matrix not implemented")

            edges_positional = self.edges_positional
            return csr_array(
                (data, (edges_positional[:, 0], edges_positional[:, 1])), shape=(n, n)
            )
