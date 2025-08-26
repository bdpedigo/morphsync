from functools import partial
from typing import Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from .base import FacetFrame
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
    def layer_names(self) -> list[str]:
        return list(self._layers.keys())

    def has_layer(self, name) -> bool:
        return name in self._layers

    def get_layer(self, name) -> FacetFrame:
        if name not in self._layers:
            raise KeyError(f"Layer '{name}' does not exist.")
        return self._layers[name]

    @property
    def layer_types(self) -> dict:
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
            # TODO delete links?

    def add_mesh(self, mesh, name: str, copy=True, **kwargs) -> None:
        native_mesh = mesh
        mesh = Mesh(native_mesh, copy=copy, **kwargs)
        self._add_layer(mesh, name)

    def add_points(self, points, name: str, copy=True, **kwargs) -> None:
        native_points = points
        points = Points(native_points, copy=copy, **kwargs)
        self._add_layer(points, name)

    def add_graph(self, graph, name: str, copy=True, **kwargs) -> None:
        native_graph = graph
        graph = Graph(native_graph, copy=copy, **kwargs)
        self._add_layer(graph, name)

    def add_table(
        self, dataframe: pd.DataFrame, name: str, copy=True, **kwargs
    ) -> None:
        table = Table(dataframe, copy=copy, **kwargs)
        self._add_layer(table, name)

    def add_layer(self, data, name: str, layer_type: str, copy=True, **kwargs) -> None:
        if layer_type == "mesh":
            self.add_mesh(data, name, copy=copy, **kwargs)
        elif layer_type == "points":
            self.add_points(data, name, copy=copy, **kwargs)
        elif layer_type == "graph":
            self.add_graph(data, name, copy=copy, **kwargs)
        elif layer_type == "table":
            self.add_table(data, name, copy=copy, **kwargs)
        else:
            raise ValueError(
                "`layer_type` must be one of 'points', 'graph', 'mesh', or 'table'"
            )

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
            if not {source, target}.issubset(mapping.columns):
                raise ValueError(
                    f"Mapping DataFrame must have columns '{source}' and '{target}'"
                )
            mapping_type = "specified"
            mapping_df = mapping[[source, target]].copy()
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
        for node in self._layers.keys():
            if node not in link_graph:
                link_graph.add_node(node)
        return link_graph

    def get_link_path(self, source, target):
        return nx.shortest_path(self.link_graph, source, target)

    def get_mapping_paths(
        self,
        source: str,
        target: str,
        source_index: Optional[Union[np.ndarray, pd.Index]] = None,
        validate=None,
    ) -> pd.DataFrame:
        """
        Find mappings from source to target layers using the entire link graph, and
        describe the mapping at each step.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.
        source_index : pd.Index, optional
            Index of the source layer to map from. If None, uses all indices in the
            source layer.
        validate : str, optional
            Whether to validate the mapping at each step. If specified, checks if each
            mapping between layers is of the specified type. Options are:
            - “one_to_one” or “1:1”: check if join keys are unique in both source and target layers.
            - “one_to_many” or “1:m”: check if join keys are unique in the source dataset.
            - “many_to_one” or “m:1”: check if join keys are unique in the target dataset.
            - “many_to_many” or “m:m”: allowed, but does not result in checks.

        Returns
        -------
        :
            Mapped indices in the target layer corresponding to the source_index in the
            source layer. Note that there may be duplicates or -1 values if the
            mapping is not one-to-one. -1 denotes null or no mapping.
        """
        # TODO: make this operate on the mappings themselves and then only apply to
        # the index at the end
        if source_index is None:
            source_index = self._layers[source].nodes_index
        else:
            if not isinstance(source_index, pd.Index):
                source_index = pd.Index(source_index)

        joined_mapping = pd.DataFrame(index=source_index)
        joined_mapping[source] = source_index
        for current_source, current_target in nx.utils.pairwise(
            self.get_link_path(source, target)
        ):
            mapping_series = (
                self._links[(current_source, current_target)]
                .set_index(current_source)[current_target]
                .drop(
                    -1, errors="ignore"
                )  # not sure if necessary but being safe for now
            )
            # catch pandas MergeError due to validate here, raise a more informative error

            try:
                joined_mapping = joined_mapping.join(
                    mapping_series, how="left", on=current_source, validate=validate
                )
            except pd.errors.MergeError as e:
                raise pd.errors.MergeError(
                    f"Mapping from '{current_source}' to '{current_target}' failed validation '{validate}'."
                ) from e

            # TODO decide whether to use -1 or NaN for unmapped
            joined_mapping[current_target] = (
                joined_mapping[current_target].fillna(-1).astype(int)
            )
        joined_mapping = joined_mapping.loc[source_index]
        return joined_mapping

    def get_mapping(
        self, source: str, target: str, source_index=None, validate=None, dropna=True
    ) -> pd.Series:
        """
        Find mappings from source to target layers using the entire link graph.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.
        source_index : pd.Index, optional
            Index of the source layer to map from. If None, uses all indices in the
            source layer.
        validate : str, optional
            Whether to validate the mapping at each step. If specified, checks if each
            mapping between layers is of the specified type. Options are:
            - “one_to_one” or “1:1”: check if join keys are unique in both source and target layers.
            - “one_to_many” or “1:m”: check if join keys are unique in the source dataset.
            - “many_to_one” or “m:1”: check if join keys are unique in the target dataset.
            - “many_to_many” or “m:m”: allowed, but does not result in checks.
        dropna : bool, default True
            Whether to drop -1/null values (no mapping) from the output.

        Returns
        -------
        :
            Series with nodes in the source layer as the index and mapped nodes in
            the target layer as the values. Note that there may be duplicates or -1
            values if the mapping is not one-to-one. -1 denotes null or no mapping.

        Notes
        -----
        This function is a convenience wrapper around `get_mapping_path` that returns
        just the final mapping as a Series. If you need to see the full mapping at
        each step, use `get_mapping_path`.
        """
        mapping_path = self.get_mapping_paths(source, target, source_index, validate)
        mapping = mapping_path.set_index(source)[target]
        if dropna:
            mapping = mapping[mapping != -1]
        return mapping

    def get_masking(self, source, target, source_index=None):
        """Gets any elements from another layer that map to any of source_index."""

        mapping = self.get_mapping(source, target, source_index, dropna=True)
        target_ids = mapping.values
        target_ids = np.unique(target_ids)
        return target_ids

    def get_mapped_nodes(
        self, source, target, source_index=None, replace_index=True, validate=None
    ):
        """
        Get features from the target layer, for specified nodes mapped from the source
        layer.
        """
        if source_index is None:
            source_index = self._layers[source].nodes_index

        mapping = self.get_mapping(
            source, target, source_index, validate=validate, dropna=True
        )
        target_index = mapping.values
        out = self._layers[target].nodes.reindex(target_index)
        if replace_index:
            out = out.set_index(mapping.index)
        return out

    def assign_from_mapping(
        self,
        source: str,
        target: str,
        columns: Union[str, list, dict],
    ):
        """
        Assign values from the source layer to the target layer based on the mapping.
        """
        if isinstance(columns, dict):
            target_columns = list(columns.keys())
            source_columns = list(columns.values())
        elif isinstance(columns, str):
            target_columns = [columns]
            source_columns = [columns]
        elif isinstance(columns, list):
            target_columns = columns
            source_columns = columns
        else:
            raise TypeError("Columns must be a str, list, or dict")
        mapped_nodes = self.get_mapped_nodes(
            source, target, replace_index=True, validate="m:1"
        )[target_columns]
        source_layer = self.get_layer(source)
        source_layer.nodes[source_columns] = mapped_nodes

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
        new_morphology._links = self._links

        return new_morphology

    def get_link_as_layer(self, source, target):
        mapping = self.get_mapping(source, target)
        source_index = mapping.index
        target_index = mapping.values
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

    def query_nodes(self, layer_name, query_str):
        layer_query = self._layers[layer_name].query_nodes(query_str)
        new_index = layer_query.nodes.index
        return self._generate_new_morphology(layer_name, new_index)
