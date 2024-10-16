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

    @property
    def edges(self):
        return self.facets.values

    @property
    def edges_df(self):
        return self.facets

    def to_pyvista(self):
        pv = import_pyvista()
        lines = np.empty((len(self.edges), 3), dtype=int)
        lines[:, 0] = 2
        lines[:, 1:3] = self.edges

        return pv.PolyData(self.vertices, lines=lines)
