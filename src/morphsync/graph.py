import numpy as np

from .base import FacetFrame


class Graph(FacetFrame):
    def __init__(self, graph, *args, **kwargs):
        if hasattr(graph, "vertices") and hasattr(graph, "edges"):
            vertices = graph.vertices
            edges = graph.edges
        elif isinstance(graph, tuple):
            vertices, edges = graph
        super().__init__(vertices, edges, *args, **kwargs)
        self._graph = graph

    def __repr__(self):
        return f"Graph(nodes={self.nodes.shape}, edges={self.edges.shape})"

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def edges(self):
        return self.edges_df.values

    @property
    def edges_df(self):
        return (
            self.facets[self.relation_columns]
            if self.relation_columns is not None
            else self.facets
        )

    @property
    def edges_positional(self):
        return np.vectorize(self.nodes.index.get_loc)(self.edges)

    def to_adjacency(self, return_as="csr", weights=None, symmetrize=False):
        if return_as == "csr":
            from scipy.sparse import csr_array

            n = self.n_nodes
            if weights is None:
                data = np.ones(self.n_edges)
            else:
                data = self.facets[weights].values

            edges_positional = self.edges_positional

            if symmetrize:
                # NOTE: this would add edges if in both directions
                edges_positional = np.concatenate(
                    [edges_positional, edges_positional[:, ::-1]], axis=0
                )
                data = np.concatenate([data, data], axis=0)

            return csr_array(
                (data, (edges_positional[:, 0], edges_positional[:, 1])), shape=(n, n)
            )

    @property
    def is_spatially_valid(self):
        is_valid = self.vertices.shape[1] == 3 and self.vertices.shape[0] > 0
        is_valid &= self.edges.shape[1] == 2 and self.edges.shape[0] > 0
        return is_valid
