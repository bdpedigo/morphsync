from functools import partial

import networkx as nx
import numpy as np
import pandas as pd

from .graph import Graph
from .mapping import project_points_to_nearest
from .mesh import Mesh
from .points import Points
from .table import Table


class MorphSync:
    def __init__(self, name=None):
        self.name = name
        self._layers = {}
        self._links = {}
        self._link_origins = {}
        self._delayed_add_links = {}

    def __repr__(self):
        return f"MorphLink(name={self.name}, layers={list(self._layers.keys())})"

    def get_params(self):
        return {
            "name": self.name,
        }

    @property
    def layer_names(self):
        return list(self._layers.keys())

    @property
    def layer_types(self):
        return {name: layer.__class__ for name, layer in self._layers.items()}

    def _add_layer(self, layer, name):
        self._layers[name] = layer
        self.__setattr__(name, layer)
        for (
            delayed_source,
            delayed_target,
        ), add_link_func in self._delayed_add_links.items():
            if delayed_source == name or delayed_target == name:
                add_link_func()

    def drop_layer(self, name):
        if name in self._layers:
            del self._layers[name]
            delattr(self, name)
            # TODO delete links

    def add_mesh(self, mesh, name: str, **kwargs) -> None:
        native_mesh = mesh
        mesh = Mesh(native_mesh, **kwargs)
        self._add_layer(mesh, name)

    def add_points(self, points, name: str, **kwargs) -> None:
        native_points = points
        points = Points(native_points, **kwargs)
        self._add_layer(points, name)

    def add_graph(self, graph, name: str, **kwargs) -> None:
        native_graph = graph
        graph = Graph(native_graph, **kwargs)
        self._add_layer(graph, name)

    def add_table(self, dataframe: pd.DataFrame, name: str, **kwargs) -> None:
        table = Table(dataframe, **kwargs)
        self._add_layer(table, name)

    def add_point_annotations(
        self,
        point_annotations: pd.DataFrame,
        spatial_columns: list,
        name: str,
        annotations_suffix: str = "_annotations",
        **kwargs,
    ) -> None:
        """
        Add point annotations as a new point layer and new table with a link to
        those points.
        """
        spatial_points = point_annotations[spatial_columns]
        non_spatial_columns = point_annotations.columns.difference(spatial_columns)
        annotations = point_annotations[non_spatial_columns]
        self.add_points(spatial_points, name, **kwargs)
        self.add_table(
            annotations,
            name + annotations_suffix,
        )
        self.add_link(name, name + annotations_suffix, mapping="index", reciprocal=True)

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

    def add_link(self, source, target, mapping="closest", reciprocal=True):
        # TODO
        # raise TypeError(
        #     "mapping must be a str, np.ndarray, pd.DataFrame, pd.Series, pd.Index, or dict"
        # )
        if source not in self._layers or target not in self._layers:
            delayed_add_link = partial(
                self.add_link, source, target, mapping, reciprocal
            )
            self._delayed_add_links[(source, target)] = delayed_add_link
            return

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
                mapping_df = pd.DataFrame(index=np.arange(len(mapping)))
                mapping_df[source] = source_layer.points_index.values
                mapping_df[target] = mapping
                # NOTE: old code, this was messing up types
                # mapping_df = pd.DataFrame(
                #     data=np.stack((source_layer.points_index, mapping), axis=1),
                #     columns=[source, target],
                # )

            else:
                raise NotImplementedError()
        elif isinstance(mapping, pd.DataFrame):
            raise NotImplementedError()
        elif isinstance(mapping, pd.Series):
            mapping_type = "specified"
            # if (mapping.index.name is not None) and mapping.index.name != source:
            #     raise UserWarning(
            #         f"Index name of mapping Series ({mapping.index.name}) does not match source layer name ({source})."
            #     )
            # if mapping.name is not None and mapping.name != target:
            #     raise UserWarning(
            #         f"Name of mapping Series ({mapping.name}) does not match target layer name ({target})."
            #     )
            mapping_df = mapping.to_frame().reset_index()
            mapping_df.columns = [source, target]

        elif isinstance(mapping, pd.Index):
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
        # TODO: make this operate on the mappings themselves and then only apply to
        # the index at the end
        if source_index is None:
            source_index = self._layers[source].nodes_index
        else:
            if not isinstance(source_index, pd.Index):
                source_index = pd.Index(source_index)
        current_index = source_index.values.copy()
        for current_source, current_target in nx.utils.pairwise(
            self.get_link_path(source, target)
        ):
            mapping_series = self._links[(current_source, current_target)].set_index(
                current_source
            )[current_target]
            # print(mapping_df)
            current_index = mapping_series.reindex(current_index, fill_value=-1).values
            # print(mapping)
            # mapping = mapping.values

            # # TODO total hack to preserve -1s, "preserve_missing_labels" in fastremap
            # # was not working
            # mapping = np.concatenate((mapping, np.array([[-1, -1]])), axis=0)

            # current_index = fastremap.remap_from_array_kv(
            #     current_index,
            #     mapping[:, 0],
            #     mapping[:, 1],
            # )

        return current_index

    def get_masking(self, source, target, source_index=None):
        """Gets any elements from another layer that map to any of source_index."""
        if source_index is None:
            source_index = self._layers[source].nodes_index
        else:
            if not isinstance(source_index, pd.Index):
                source_index = pd.Index(source_index)

        current_index = source_index.values.copy()
        current_index = np.unique(current_index)
        for current_source, current_target in nx.utils.pairwise(
            self.get_link_path(source, target)
        ):
            mapping_series = self._links[(current_source, current_target)].set_index(
                current_source
            )[current_target]

            # NOTE: this is somewhat slow but not as bad as the loc
            intersection = mapping_series.index.intersection(current_index)

            # NOTE this is the slow part
            current_index = mapping_series.loc[intersection].values

            current_index = np.unique(current_index)

        return current_index

    def get_mapped_nodes(self, source, target, source_index=None, replace_index=False):
        """
        Get a new layer that is mapped from the source layer to the target layer
        using the mapping defined in the links.
        """
        if source_index is None:
            source_index = self._layers[source].nodes_index

        target_index = self.get_mapping(source, target, source_index)
        out = self._layers[target].nodes.loc[target_index]
        if replace_index:
            out = out.set_index(source_index)
        return out

    def apply_mask(self, layer_name, mask):
        layer = self._layers[layer_name]
        new_index = layer.nodes.index[mask]
        return self._generate_new_morphology(layer_name, new_index)

    def apply_mask_by_node_index(self, layer_name, new_index):
        return self._generate_new_morphology(layer_name, new_index)

    def _generate_new_morphology(self, layer_name, new_index):
        new_morphology = self.__class__(**self.get_params())
        new_morphology._add_layer(
            self._layers[layer_name].mask_by_node_index(new_index), layer_name
        )
        for other_layer_name, other_layer in self._layers.items():
            if other_layer_name != layer_name:
                other_indices = self.get_masking(
                    layer_name, other_layer_name, source_index=new_index
                )
                new_morphology._add_layer(
                    other_layer.mask_by_node_index(other_indices), other_layer_name
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
