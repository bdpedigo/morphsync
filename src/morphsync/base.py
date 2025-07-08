from typing import Union
import numpy as np
import pandas as pd
import fastremap
from cachetools import LRUCache, cached
from joblib import hash


def mask_and_remap(
    arr: np.ndarray,
    mask: Union[np.ndarray, list],
):
    """Given an array in unmasked indexing and a mask,
    return the array in remapped indexing and omit rows with masked values.

    Parameters
    ----------
    arr : np.ndarray
        NxM array of indices
    mask : Union[np.ndarray, list
        1D array of indices to mask, either as a boolean mask or as a list of indices
    """
    if np.array(mask).dtype == bool:
        mask = np.where(mask)[0]
    return _mask_and_remap(np.array(arr, dtype=int), mask)


def _numpy_hash(*args, **kwargs):
    return tuple(hash(x) for x in args) + tuple(hash(x) for x in kwargs.items())


@cached(cache=LRUCache(maxsize=128), key=_numpy_hash)
def _mask_and_remap(
    arr: np.ndarray,
    mask: np.ndarray,
):
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


class FacetFrame:
    # a base class for representing a facet complex in a 3D space
    # a facet complex with no facets is a point cloud
    # a facet complex with edges is a spatial network/graph
    # a facet complex with faces (3-facets i.e. triangles) is a mesh
    def __init__(self, nodes, facets, spatial_columns=None):
        if spatial_columns is None:
            spatial_columns = ["x", "y", "z"]
        if not isinstance(nodes, pd.DataFrame):
            if isinstance(nodes, np.ndarray):
                if nodes.shape[1] == 3:
                    nodes = pd.DataFrame(nodes, columns=spatial_columns)
                else:
                    raise ValueError("Nodes must be a 3D array")
            else:
                raise ValueError("Nodes must be a DataFrame or a 3D array")
        self.nodes: pd.DataFrame = nodes
        if facets is None:
            facets = pd.DataFrame()
        if not isinstance(facets, pd.DataFrame):
            facets = pd.DataFrame(facets)
        self.facets: pd.DataFrame = facets
        self.spatial_columns = spatial_columns

    def get_params(self):
        return {
            "spatial_columns": self.spatial_columns,
        }

    @property
    def vertices(self):
        """Array of the spatial coordinates of the vertices"""
        return self.vertices_df.values

    @property
    def vertices_df(self):
        """DataFrame of the spatial coordinates of the vertices"""
        return self.nodes[self.spatial_columns]

    @property
    def points(self):
        """Alias for vertices"""
        return self.vertices

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_vertices(self):
        return self.n_nodes

    @property
    def n_points(self):
        return self.n_nodes

    @property
    def nodes_index(self):
        return self.nodes.index

    @property
    def vertices_index(self):
        return self.nodes_index

    @property
    def points_index(self):
        return self.nodes_index

    @property
    def facets_index(self):
        return self.facets.index

    @property
    def edge_index(self):
        return self.facets_index

    @property
    def facets_positional(self) -> np.ndarray:
        return mask_and_remap(self.facets, self.nodes.index)

    def query_nodes(self, query_str):
        new_nodes = self.nodes.query(query_str)
        new_index = new_nodes.index
        return self.mask_by_node_index(new_index, new_nodes=new_nodes)

    def mask_nodes(self, mask):
        new_nodes = self.nodes.iloc[mask]
        new_index = new_nodes.index
        return self.mask_by_node_index(new_index, new_nodes=new_nodes)

    def mask_by_node_index(self, new_index, new_nodes=None):
        if new_nodes is None:
            new_nodes = self.nodes.loc[new_index]
        new_facets = self.facets[self.facets.isin(new_index).all(axis=1)]
        return self.__class__((new_nodes, new_facets), **self.get_params())
