import networkx as nx
import numpy as np
import pandas as pd

from .graph import Graph
from .mapping import project_points_to_nearest
from .mesh import Mesh
from .points import Points
from .table import Table


class MorphSync:
    def __init__(self):
        self._layers = {}
        self._links = {}
        self._link_origins = {}

    def __repr__(self):
        return f"MorphLink(layers={list(self._layers.keys())}, links={list(self._links.keys())})"

    @property
    def layer_names(self):
        return list(self._layers.keys())

    @property
    def layer_types(self):
        return {name: layer.__class__ for name, layer in self._layers.items()}

    def _add_layer(self, layer, name):
        self._layers[name] = layer
        self.__setattr__(name, layer)

    def drop_layer(self, name):
        if name in self._layers:
            del self._layers[name]
            delattr(self, name)
            # TODO delete links

    def add_mesh(self, mesh, name) -> None:
        native_mesh = mesh
        mesh = Mesh(native_mesh)
        self._add_layer(mesh, name)

    def add_points(self, points, name, **kwargs) -> None:
        native_points = points
        points = Points(native_points, **kwargs)
        self._add_layer(points, name)

    def add_graph(self, graph, name, **kwargs) -> None:
        native_graph = graph
        graph = Graph(native_graph, **kwargs)
        self._add_layer(graph, name)

    def add_table(self, dataframe, name, **kwargs) -> None:
        table = Table(dataframe, **kwargs)
        self._add_layer(table, name)

    @property
    def layers(self) -> pd.DataFrame:
        layers = {}
        for name, layer in self._layers.items():
            layers[name] = {
                "layer": layer.__repr__(),
                "layer_type": layer.__class__.__name__,
            }

        layers = pd.DataFrame(layers).T
        layers.index.name = "name"
        return layers

    def add_link(self, source, target, mapping="closest", reciprocal=False):
        # TODO
        # raise TypeError(
        #     "mapping must be a str, np.ndarray, pd.DataFrame, pd.Series, pd.Index, or dict"
        # )

        source_layer = self._layers[source]
        target_layer = self._layers[target]
        if isinstance(mapping, str):
            mapping_type = mapping
            if mapping == "closest":
                mapping_array = project_points_to_nearest(
                    source_layer.points, target_layer.points
                )
                mapping_df = pd.DataFrame(
                    data=np.stack(
                        (
                            source_layer.points_index,
                            target_layer.points_index[mapping_array],
                        ),
                        axis=1,
                    ),
                    columns=[source, target],
                )
            elif mapping == "index":
                mapping_df = pd.DataFrame(
                    data=np.stack(
                        (
                            source_layer.points_index,
                            target_layer.points_index,
                        ),
                        axis=1,
                    ),
                    columns=[source, target],
                )
            elif mapping == "order":
                raise NotImplementedError()
        elif isinstance(mapping, np.ndarray):
            mapping_type = "specified"
            # assumes that the mapping is a 1d array where the index is on source
            # vertices, and the value is the index on target vertices
            if mapping.ndim == 1:
                mapping_df = pd.DataFrame(
                    data=np.stack((source_layer.points_index, mapping), axis=1),
                    columns=[source, target],
                )

            else:
                raise NotImplementedError()
        elif isinstance(mapping, pd.DataFrame):
            raise NotImplementedError()
        elif isinstance(mapping, (pd.Series, pd.Index)):
            raise NotImplementedError()
        elif isinstance(mapping, dict):
            mapping_type = "specified"
            mapping_df = pd.Series(mapping)
            mapping_df.index.name = source
            mapping_df.name = target
            mapping_df = mapping_df.to_frame().reset_index()

        self._links[(source, target)] = mapping_df
        self._link_origins[(source, target)] = mapping_type
        if reciprocal:
            self._links[(target, source)] = mapping_df
            self._link_origins[(target, source)] = mapping_type

    def get_link(self, source, target):
        return self._links[(source, target)]

    @property
    def links(self):
        rows = []
        for (source, target), link in self._links.items():
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "link": link,
                    "link_origin": self._link_origins[(source, target)],
                }
            )

        links = pd.DataFrame(rows).set_index(["source", "target"])
        return links

    @property
    def link_graph(self) -> nx.DiGraph:
        link_graph = nx.DiGraph()
        for (source, target), link in self._links.items():
            link_graph.add_edge(source, target)
        return link_graph

    def get_link_path(self, source, target):
        return nx.shortest_path(self.link_graph, source, target)

    def get_mapping(self, source, target, source_index=None):
        # # TODO: make this operate on the graph of mappings
        # out = self._links[(source, target)].set_index(source).loc[source_index]
        # if squeeze:
        #     return out.squeeze()
        if source_index is None:
            source_index = self._layers[source].nodes_index

        current_index = source_index
        for current_source, current_target in nx.utils.pairwise(
            self.get_link_path(source, target)
        ):
            mapping = self._links[(current_source, current_target)]
            current_index = pd.Index(
                mapping.set_index(current_source).loc[current_index][current_target]
            )

        return current_index

    def apply_mask(self, layer_name, mask):
        layer = self.layers.loc[layer_name].layer
        new_index = layer.vertices_index[mask]
        return self._generate_new_morphology(layer_name, new_index)

    def _generate_new_morphology(self, layer_name, new_index):
        new_morphology = self.__class__()
        new_morphology._add_layer(
            self._layers[layer_name].mask_by_vertex_index(new_index), layer_name
        )
        for other_layer_name, other_layer in self._layers.items():
            if other_layer_name != layer_name:
                other_indices = self.get_mapping(other_layer_name, layer_name)
                new_morphology._add_layer(
                    other_layer.mask_by_vertex_index(other_indices), other_layer_name
                )
        return new_morphology

    def get_link_as_layer(self, source, target):
        source_index = self._layers[source].nodes_index
        target_index = self.get_mapping(source, target)
        source_nodes = self._layers[source].nodes.loc[source_index]
        target_nodes = self._layers[target].nodes.loc[target_index]
        node_positions = np.concatenate(
            [source_nodes.values, target_nodes.values], axis=0
        )
        edges = pd.DataFrame(
            data=np.stack(
                (
                    np.arange(len(source_index)),
                    len(source_index) + np.arange(len(target_index)),
                ),
                axis=1,
            ),
        )
        return Graph((node_positions, edges))

    def query_nodes(self, query_str, layer_name):
        layer_query = self._layers[layer_name].query_nodes(query_str)
        new_index = layer_query.nodes.index
        return self._generate_new_morphology(layer_name, new_index)
