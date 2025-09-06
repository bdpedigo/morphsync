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
    # a facet complex with faces (3-facets i.e. triangles) is a mesh, etc.
    def __init__(
        self,
        nodes,
        facets,
        spatial_columns: Optional[list] = None,
        relation_columns: Optional[list] = None,
        inplace_data: bool = False,
    ):
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
            # if spatial_columns is None:
            #     if nodes.shape[1] != 3:
            #         raise ValueError(
            #             "If spatial_columns is not provided, nodes must have 3 columns"
            #         )
        if not inplace_data:
            self.nodes: pd.DataFrame = nodes.copy()
        else:
            self.nodes = nodes

        if spatial_columns is None:
            spatial_columns = []
        self.spatial_columns = spatial_columns

        if facets is None:
            facets = pd.DataFrame()
        if not isinstance(facets, pd.DataFrame):
            facets = pd.DataFrame(facets)
            if relation_columns is None:
                relation_columns = facets.columns.tolist()
        if inplace_data:
            self.facets: pd.DataFrame = facets
        else:
            self.facets = facets.copy()

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

    # @property
    # def points_df(self):
    #     """Alias for vertices_df"""
    #     return self.vertices_df

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_vertices(self) -> int:
        return self.n_nodes

    @property
    def n_points(self) -> int:
        return self.n_nodes

    @property
    def n_facets(self) -> int:
        return len(self.facets)

    @property
    def nodes_index(self) -> pd.Index:
        return self.nodes.index

    @property
    def vertices_index(self) -> pd.Index:
        return self.nodes_index

    @property
    def points_index(self) -> pd.Index:
        return self.nodes_index

    @property
    def facets_index(self) -> pd.Index:
        return self.facets.index

    @property
    def edge_index(self) -> pd.Index:
        return self.facets_index

    @property
    def facets_positional(self) -> np.ndarray:
        return mask_and_remap(self.facets[self.relation_columns], self.nodes.index)

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
            new_nodes = self.nodes.loc[self.nodes.index.intersection(new_index)]
            # new_nodes = self.nodes.reindex(new_index)

        # old
        new_facets = self.facets[
            self.facets[self.relation_columns].isin(new_index).all(axis=1)
        ]

        # new  node_mapping = {k: v for v, k in enumerate(new_index)}
        # select_facets = self.facets[self.facets.isin(new_index).all(axis=1)]
        # select_facet_array = select_facets[self.relation_columns].values
        # # new_facets = fastremap.remap(select_facets, node_mapping)
        # new_facet_array = np.vectorize(
        #     lambda x: node_mapping.get(x, -1), otypes=[select_facet_array.dtype]
        # )(select_facet_array)
        # new_facets = select_facets.copy()
        # new_facets[self.relation_columns] = new_facet_array
        # print(self.get_params())
        out = self.__class__((new_nodes, new_facets), **self.get_params())
        return out

    def get_params(self):
        return {
            "spatial_columns": self.spatial_columns,
            "relation_columns": self.relation_columns,
        }
