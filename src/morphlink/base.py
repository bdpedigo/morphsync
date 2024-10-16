import numpy as np
import pandas as pd


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
    def points_df(self):
        """Alias for vertices_df"""
        return self.vertices_df

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
