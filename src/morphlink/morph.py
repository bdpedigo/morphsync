import networkx as nx
import numpy as np
import pandas as pd

from .graph import Graph
from .integration import import_pyvista
from .mapping import project_points_to_nearest
from .mesh import Mesh
from .points import Points


class MorphLink:
    def __init__(self):
        # self._meshes = {}
        # self._spatial_graphs = {}
        # self._points = {}
        self._layers = {}
        self._links = {}
        self._tables = {}

    def __repr__(self):
        return f"MorphLink(layers={list(self._layers.keys())}, links={list(self._links.keys())}, tables={list(self._tables.keys())})"

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

    def add_table(self, dataframe, name):
        self._tables[name] = dataframe
        self.__setattr__(name, dataframe)

    @property
    def layers(self) -> pd.DataFrame:
        rows = []
        for name, layer in self._layers.items():
            rows.append(
                {"name": name, "layer": layer, "layer_type": layer.__class__.__name__}
            )

        layers = pd.DataFrame(rows).set_index("name")
        return layers

    def add_link(self, source, target, mapping="closest", link_type=None):
        source_layer = self._layers[source]
        target_layer = self._layers[target]
        if isinstance(mapping, str):
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
                self._links[(source, target)] = mapping_df
            elif mapping == "index":
                raise NotImplementedError()
            elif mapping == "order":
                raise NotImplementedError()
        elif isinstance(mapping, np.ndarray):
            # assumes that the mapping is a 1d array where the index is on source
            # vertices, and the value is the index on target vertices
            if mapping.ndim == 1:
                mapping_df = pd.DataFrame(
                    data=np.stack((source_layer.points_index, mapping), axis=1),
                    columns=[source, target],
                )
                self._links[(source, target)] = mapping_df
            else:
                raise NotImplementedError()
        elif isinstance(mapping, pd.DataFrame):
            raise NotImplementedError()
        elif isinstance(mapping, (pd.Series, pd.Index)):
            raise NotImplementedError()
        elif isinstance(mapping, dict):
            raise NotImplementedError()
        else:
            raise TypeError(
                "mapping must be a str, np.ndarray, pd.DataFrame, pd.Series, pd.Index, or dict"
            )

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
                    # "link_type": link.__class__.__name__,
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

    # @property
    # def link_graph(self) -> nx.Graph:
    #     link_graph = nx.Graph()
    #     for (source, target), link in self._links.items():
    #         link_graph.add_edge(source, target)
    #     return link_graph

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

    def query_nodes(self, query_str, layer_name):
        layer_query = self._layers[layer_name].query_nodes(query_str)
        new_index = layer_query.nodes.index
        new_morphology = self.__class__()
        for other_layer_name, other_layer in self._layers.items():
            other_indices = self.get_mapping(other_layer_name, layer_name)
            mask = other_indices.isin(new_index)
            new_other_layer = other_layer.mask_nodes(mask)
            new_morphology._add_layer(new_other_layer, other_layer_name)
            # TODO drop links that are no longer valid
        return new_morphology

    def to_pyvista(self) -> dict["pv.PolyData"]:
        pv = import_pyvista()
        return {name: layer.to_pyvista() for name, layer in self._layers.items()}
