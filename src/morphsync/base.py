from typing import Optional, Union

import fastremap
import numpy as np
import pandas as pd
from cachetools import LRUCache, cached
from joblib import hash

DEFAULT_SPATIAL_COLUMNS = ["x", "y", "z"]


def mask_and_remap(
    arr: np.ndarray,
    mask: Union[np.ndarray, list],
) -> np.ndarray:
    """Given an array in unmasked indexing and a mask,
    return the array in remapped indexing and omit rows with masked values.

    Parameters
    ----------
    arr :
        NxM array of indices
    mask :
        1D array of indices to mask, either as a boolean mask or as a list of indices
    """
    if np.array(mask).dtype == bool:
        mask = np.where(mask)[0]
    return _mask_and_remap(np.array(arr, dtype=int), mask)


def _numpy_hash(*args, **kwargs) -> tuple:
    return tuple(hash(x) for x in args) + tuple(hash(x) for x in kwargs.items())


@cached(cache=LRUCache(maxsize=128), key=_numpy_hash)
def _mask_and_remap(
    arr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    mask_dict = {k: v for k, v in zip(mask, range(len(mask)))}
    mask_dict[-1] = -1

    arr_offset = arr + 1
    arr_mask_full = fastremap.remap(
        fastremap.mask_except(arr_offset, list(mask + 1)) - 1,
        mask_dict,
    )
    if len(arr_mask_full.shape) == 1:
        return arr_mask_full[~np.any(arr_mask_full == -1)]
    else:
        return arr_mask_full[~np.any(arr_mask_full == -1, axis=1)]


class Layer:
    def __init__(
        self,
        nodes: Union[pd.DataFrame, np.ndarray],
        facets: Union[pd.DataFrame, np.ndarray],
        spatial_columns: Optional[list] = None,
        relation_columns: Optional[list] = None,
        copy: bool = True,
        **kwargs,
    ):
        """A class for representing a set of nodes and their relationships (facets).

        Parameters
        ----------
        nodes :
            A DataFrame or nx3 array of the nodes/vertices/points in the layer.
            If an array is provided, it must be nx3 and the columns will be named
            "x", "y", and "z" unless `spatial_columns` is provided.
        facets :
            A DataFrame or m x k array of the facets/edges/faces in the layer.
            Some columns must correspond to indices in `nodes`, these are specified
            by `relation_columns`.
        spatial_columns :
            A list of column names in `nodes` that correspond to spatial coordinates.
            If not provided and `nodes` is a DataFrame, all columns will be used. If
            `nodes` is an array, defaults to ["x", "y", "z"].
        relation_columns :
            A list of column names in `facets` that correspond to indices in `nodes`.
            If not provided and `facets` is a DataFrame, all columns will be used.
        copy :
            Whether to copy the input DataFrames. If False, the input DataFrames
            may be modified in place.
        """
        if not isinstance(nodes, pd.DataFrame):
            if isinstance(nodes, np.ndarray):
                if nodes.shape[1] == 3:
                    nodes = pd.DataFrame(nodes, columns=DEFAULT_SPATIAL_COLUMNS)
                    if spatial_columns is None:
                        spatial_columns = DEFAULT_SPATIAL_COLUMNS
                else:
                    raise ValueError("Nodes must be an nx3 array")
            else:
                raise ValueError("Nodes must be a DataFrame or an nx3 array")

        if copy:
            nodes = nodes.copy()
        self.nodes: pd.DataFrame = nodes

        if spatial_columns is None:
            spatial_columns = []
        self.spatial_columns = spatial_columns

        if facets is None:
            facets = pd.DataFrame()
        if not isinstance(facets, pd.DataFrame):
            facets = pd.DataFrame(facets)
            if relation_columns is None:
                relation_columns = facets.columns.tolist()
        if copy:
            facets = facets.copy()
        self.facets: pd.DataFrame = facets

        if relation_columns is None:
            relation_columns = []
        self.relation_columns = relation_columns

    @property
    def vertices(self) -> np.ndarray:
        """Array of the spatial coordinates of the vertices"""
        return self.vertices_df.values

    @property
    def vertices_df(self) -> pd.DataFrame:
        """DataFrame of the spatial coordinates of the vertices"""
        return (
            self.nodes[self.spatial_columns]
            if self.spatial_columns is not None
            else self.nodes
        )

    @property
    def points(self) -> np.ndarray:
        """Alias for vertices"""
        return self.vertices

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the layer."""
        return len(self.nodes)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the layer. Alias for n_nodes."""
        return self.n_nodes

    @property
    def n_points(self) -> int:
        """Number of points in the layer. Alias for n_nodes."""
        return self.n_nodes

    @property
    def n_facets(self) -> int:
        """Number of facets (edges/faces) in the layer."""
        return len(self.facets)

    @property
    def nodes_index(self) -> pd.Index:
        """Index of the nodes DataFrame."""
        return self.nodes.index

    @property
    def vertices_index(self) -> pd.Index:
        """Index of the vertices. Alias for nodes_index."""
        return self.nodes_index

    @property
    def points_index(self) -> pd.Index:
        """Index of the points. Alias for nodes_index."""
        return self.nodes_index

    @property
    def facets_index(self) -> pd.Index:
        """Index of the facets DataFrame."""
        return self.facets.index

    @property
    def edge_index(self) -> pd.Index:
        """Index of the edges. Alias for facets_index."""
        return self.facets_index

    @property
    def facets_positional(self) -> np.ndarray:
        """Array of the facets in positional indexing, such that 0 corresponds to the
        first node in its current node index ordering"""
        return mask_and_remap(self.facets[self.relation_columns], self.nodes.index)

    def query_nodes(self, query_str: str):
        """Query the nodes DataFrame and return a new layer with the
        corresponding nodes and facets.

        Parameters
        ----------
        query_str :
            A query string to pass to `pd.DataFrame.query` on the nodes DataFrame.

        Returns
        -------
        :
            A new layer with the queried nodes and corresponding facets.

        Notes
        -----
        When masking by nodes, only relationships that reference exclusively the
        remaining nodes are kept.
        """
        new_nodes = self.nodes.query(query_str)
        new_index = new_nodes.index
        return self.mask_by_node_index(new_index, new_nodes=new_nodes)

    def mask_nodes(self, mask: np.ndarray):
        """Mask the nodes DataFrame and return a new layer with the
        corresponding nodes and facets.

        Parameters
        ----------
        mask :
            A boolean mask array to filter the nodes DataFrame. This masking is applied
            in positional indexing (i.e. order, not key matters).

        Returns
        -------
        :
            A new layer with the masked nodes and corresponding facets.

        Notes
        -----
        When masking by nodes, only relationships that reference exclusively the
        remaining nodes are kept.
        """
        new_nodes = self.nodes.iloc[mask]
        new_index = new_nodes.index
        return self.mask_by_node_index(new_index, new_nodes=new_nodes)

    def mask_by_node_index(
        self,
        new_index: Union[np.ndarray, pd.Index, pd.Series],
        new_nodes: Optional[pd.DataFrame] = None,
    ):
        """Create a new layer containing only the specified nodes and their facets.

        Parameters
        ----------
        new_index :
            Index of nodes to keep in the new layer.
        new_nodes :
            Pre-filtered nodes DataFrame. If None, nodes will be filtered automatically
            based on new_index.

        Returns
        -------
        :
            A new layer instance containing only the specified nodes and facets that
            reference those nodes.

        Notes
        -----
        Only facets that reference exclusively the nodes in new_index are kept.
        """
        if new_nodes is None:
            new_nodes = self.nodes.loc[self.nodes.index.intersection(new_index)]
            # new_nodes = self.nodes.reindex(new_index)

        # old
        new_facets = self.facets[
            self.facets[self.relation_columns].isin(new_index).all(axis=1)
        ]
        out = self.__class__((new_nodes, new_facets), **self.get_params())
        return out

    @property
    def layer_type(self) -> str:
        """String identifier of the layer type (e.g., 'mesh', 'points', 'graph')."""
        return str(self.__class__).strip(">'").split(".")[-1].lower()

    def get_params(self) -> dict:
        """Get the parameters used to initialize this layer.

        Returns
        -------
        :
            Dictionary containing layer initialization parameters.
        """
        return {
            "spatial_columns": self.spatial_columns,
            "relation_columns": self.relation_columns,
        }
