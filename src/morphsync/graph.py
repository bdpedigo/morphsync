from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .base import Layer


class Graph(Layer):
    def __init__(
        self, graph: Union[tuple[np.ndarray, np.ndarray], Any], *args, **kwargs
    ):
        """Initialize a Graph layer.

        Parameters
        ----------
        graph :
            Either an object with 'vertices' and 'edges' attributes, or a tuple
            of (vertices, edges).
        *args : tuple
            Additional arguments passed to the parent Layer class.
        **kwargs : dict
            Additional keyword arguments passed to the parent Layer class.
        """
        if hasattr(graph, "vertices") and hasattr(graph, "edges"):
            vertices = graph.vertices
            edges = graph.edges
        elif isinstance(graph, tuple):
            vertices, edges = graph
        super().__init__(vertices, edges, *args, **kwargs)
        self._graph = graph

    def __repr__(self) -> str:
        """Return a string representation of the Graph."""
        return f"Graph(nodes={self.nodes.shape}, edges={self.edges.shape})"

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    @property
    def edges(self) -> np.ndarray:
        """Edges as a numpy array of shape (n_edges, 2)."""
        return self.edges_df.values

    @property
    def edges_df(self) -> pd.DataFrame:
        """Edges as a DataFrame containing node indices."""
        return (
            self.facets[self.relation_columns]
            if self.relation_columns is not None
            else self.facets
        )

    @property
    def edges_positional(self) -> np.ndarray:
        """Edges in positional indexing."""
        return np.vectorize(self.nodes.index.get_loc)(self.edges)

    def to_adjacency(
        self,
        return_as: str = "csr",
        weights: Optional[str] = None,
        symmetrize: bool = False,
    ):
        """Convert the graph to an adjacency matrix.

        Parameters
        ----------
        return_as :
            Format to return the adjacency matrix. Currently only "csr" is supported.
        weights :
            Column name in facets to use as edge weights. If None, uses unit weights.
        symmetrize :
            If True, add reverse edges to make the graph symmetric.

        Returns
        -------
        :
            Sparse adjacency matrix of shape (n_nodes, n_nodes).
        """
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
        else:
            raise ValueError(f"Unsupported return_as format {return_as}")

    @property
    def is_spatially_valid(self) -> bool:
        """Check if the graph has valid spatial structure.

        Returns
        -------
        :
            True if vertices are 3D and both vertices and edges are non-empty.
        """
        is_valid = self.vertices.shape[1] == 3 and self.vertices.shape[0] > 0
        is_valid &= self.edges.shape[1] == 2 and self.edges.shape[0] > 0
        return is_valid
