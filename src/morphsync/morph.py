from functools import partial
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd

from .base import Layer
from .graph import Graph
from .mapping import project_points_to_nearest
from .mesh import Mesh
from .points import Points
from .table import Table


class MorphSync:
    def __init__(self, name: Optional[Any] = None):
        """Initialize a MorphSync container for multi-layer morphological data.

        Parameters
        ----------
        name :
            Name identifier for this morphology container.
        """
        self.name = name
        self.layers = {}
        self.links = {}
        self._delayed_add_links = {}

    def __repr__(self) -> str:
        """Return a string representation of the MorphSync."""
        return f"MorphSync(name={self.name}, layers={list(self.layers.keys())})"

    def get_params(self) -> dict:
        """Get the parameters used to initialize this MorphSync container.

        Returns
        -------
        :
            Dictionary containing initialization parameters.
        """
        return {
            "name": self.name,
        }

    @property
    def layer_names(self) -> list[str]:
        """List of all layer names in the container."""
        return list(self.layers.keys())

    def has_layer(self, name: str) -> bool:
        """Check if a layer with the given name exists.

        Parameters
        ----------
        name :
            Name of the layer to check.

        Returns
        -------
        :
            True if the layer exists, False otherwise.
        """
        return name in self.layers

    def get_layer(self, name: str) -> Layer:
        """Get a layer by name.

        Parameters
        ----------
        name :
            Name of the layer to retrieve.

        Returns
        -------
        :
            The requested layer.

        Raises
        ------
        KeyError
            If the layer does not exist.
        """
        if name not in self.layers:
            raise KeyError(f"Layer '{name}' does not exist.")
        return self.layers[name]

    @property
    def layer_types(self) -> dict:
        """Dictionary mapping layer names to their class types."""
        return {name: layer.__class__ for name, layer in self.layers.items()}

    def _add_layer(self, name: str, layer: Layer) -> None:
        if not isinstance(name, str):
            raise ValueError("Layer name must be a string.")
        self.layers[name] = layer
        self.__setattr__(name, layer)
        for (
            delayed_source,
            delayed_target,
        ), add_link_func in self._delayed_add_links.items():
            if delayed_source == name or delayed_target == name:
                add_link_func()

    def drop_layer(self, name: str) -> None:
        """Remove a layer from the MorphSync.

        Parameters
        ----------
        name :
            Name of the layer to remove.
        """
        if name in self.layers:
            del self.layers[name]
            delattr(self, name)
            # TODO delete links or not?

    def add_mesh(self, name: str, mesh, copy: bool = True, **kwargs) -> None:
        """Add a mesh layer to the MorphSync.

        Parameters
        ----------
        name :
            Name for the new mesh layer.
        mesh :
            Mesh data - either an object with 'vertices' and 'faces' attributes
            or a tuple of (vertices, faces).
        copy :
            Whether to copy the input data.
        **kwargs :
            Additional arguments passed to the Mesh constructor.
        """
        native_mesh = mesh
        mesh = Mesh(native_mesh, copy=copy, **kwargs)
        self._add_layer(name, mesh)

    def add_points(self, name: str, points, copy: bool = True, **kwargs) -> None:
        """Add a points layer to the MorphSync.

        Parameters
        ----------
        name :
            Name for the new points layer.
        points :
            Point coordinates as an array, DataFrame, or tuple containing points.
        copy :
            Whether to copy the input data.
        **kwargs :
            Additional arguments passed to the Points constructor.
        """
        native_points = points
        points = Points(native_points, copy=copy, **kwargs)
        self._add_layer(name, points)

    def add_graph(self, name: str, graph, copy: bool = True, **kwargs) -> None:
        """Add a graph layer to the MorphSync.

        Parameters
        ----------
        name :
            Name for the new graph layer.
        graph :
            Graph data - either an object with 'vertices' and 'edges' attributes
            or a tuple of (vertices, edges).
        copy :
            Whether to copy the input data.
        **kwargs :
            Additional arguments passed to the Graph constructor.
        """
        native_graph = graph
        graph = Graph(native_graph, copy=copy, **kwargs)
        self._add_layer(name, graph)

    def add_table(
        self, name: str, table: pd.DataFrame, copy: bool = True, **kwargs
    ) -> None:
        """Add a table layer to the MorphSync.

        Parameters
        ----------
        name :
            Name for the new table layer.
        table :
            DataFrame containing tabular data.
        copy :
            Whether to copy the input data.
        **kwargs :
            Additional arguments passed to the Table constructor.
        """
        table = Table(table, copy=copy, **kwargs)
        self._add_layer(name, table)

    def add_layer(
        self, name: str, data: Any, layer_type: str, copy: bool = True, **kwargs
    ) -> None:
        """Add a layer of the specified type to the MorphSync.

        Parameters
        ----------
        name :
            Name for the new layer.
        data :
            Data for the layer, format depends on layer_type.
        layer_type :
            Type of layer to create. Must be one of 'mesh', 'points', 'graph', or 'table'.
        copy :
            Whether to copy the input data.
        **kwargs :
            Additional arguments passed to the layer constructor.

        Raises
        ------
        ValueError
            If layer_type is not one of the supported types.
        """
        if layer_type == "mesh":
            self.add_mesh(name, data, copy=copy, **kwargs)
        elif layer_type == "points":
            self.add_points(name, data, copy=copy, **kwargs)
        elif layer_type == "graph":
            self.add_graph(name, data, copy=copy, **kwargs)
        elif layer_type == "table":
            self.add_table(name, data, copy=copy, **kwargs)
        else:
            raise ValueError(
                "`layer_type` must be one of 'points', 'graph', 'mesh', or 'table'"
            )

    def add_link(
        self,
        source: str,
        target: str,
        mapping: Union[str, np.ndarray, pd.DataFrame, pd.Series, dict] = "closest",
    ) -> None:
        """Add a mapping link between two layers.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.
        mapping :
            Mapping specification. Options:
            - "closest": Map to nearest neighbors based on spatial coordinates
            - "index": Map by matching indices
            - np.ndarray: 1D array of target indices for each source element
            - pd.DataFrame: DataFrame with source and target columns
            - pd.Series: Series mapping source to target indices
            - dict: Dictionary mapping source to target indices
        """
        # TODO
        # raise TypeError(
        #     "mapping must be a str, np.ndarray, pd.DataFrame, pd.Series, pd.Index, or dict"
        # )
        if source not in self.layers or target not in self.layers:
            delayed_add_link = partial(self.add_link, source, target, mapping)
            self._delayed_add_links[(source, target)] = delayed_add_link
            return

        source_layer = self.layers[source]
        target_layer = self.layers[target]

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
            mapping_df = mapping[[source, target]].copy()
        elif isinstance(mapping, pd.Series):
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
            mapping_df = pd.Series(mapping)
            mapping_df.index.name = source
            mapping_df.name = target
            mapping_df = mapping_df.to_frame().reset_index()

        # TODO add a links manager, deal with reciprocal lookups properly
        # this should be a safe hack for now
        self.links[(source, target)] = mapping_df
        self.links[(target, source)] = mapping_df

    def get_link(self, source: str, target: str) -> pd.DataFrame:
        """Get the mapping DataFrame between two layers.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.

        Returns
        -------
        :
            DataFrame containing the mapping between source and target layers.
        """
        return self.links[(source, target)]

    @property
    def link_graph(self) -> nx.DiGraph:
        """NetworkX graph representing the connectivity between layers.

        Returns
        -------
        :
            Graph where nodes are layer names and edges represent mappings.
        """
        link_graph = nx.DiGraph()
        for source, target in self.links.keys():
            link_graph.add_edge(source, target)
        for node in self.layers.keys():
            if node not in link_graph:
                link_graph.add_node(node)
        return link_graph

    def get_link_path(self, source: str, target: str) -> list[str]:
        """Find the shortest path of mappings between two layers.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.

        Returns
        -------
        :
            List of layer names representing the shortest path from source to target.
        """
        return nx.shortest_path(self.link_graph, source, target)

    def get_mapping_paths(
        self,
        source: str,
        target: str,
        source_index: Optional[Union[np.ndarray, pd.Index]] = None,
        validate: Optional[str] = None,
        dropna: bool = False,
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
        source_index :
            Index of the source layer to map from. If None, uses all indices in the
            source layer.
        validate :
            Whether to validate the mapping at each step. If specified, checks if each
            mapping between layers is of the specified type. Options are:
            - "one_to_one" or "1:1": check if join keys are unique in both source and target layers.
            - "one_to_many" or "1:m": check if join keys are unique in the source dataset.
            - "many_to_one" or "m:1": check if join keys are unique in the target dataset.
            - "many_to_many" or "m:m": allowed, but does not result in checks.
        dropna :
            Whether to drop entries with null mappings. If False, returns NaN/pd.NA
            values for missing mappings.

        Returns
        -------
        :
            Mapped indices in the target layer corresponding to the source_index in the
            source layer. Format depends on `dropna` parameter.
        """
        if source_index is None:
            source_index = self.layers[source].nodes_index
        else:
            if not isinstance(source_index, pd.Index):
                source_index = pd.Index(source_index)

        joined_mapping = pd.DataFrame(index=source_index)
        joined_mapping[source] = source_index
        for current_source, current_target in nx.utils.pairwise(
            self.get_link_path(source, target)
        ):
            mapping_series = (
                self.links[(current_source, current_target)]
                .set_index(current_source)[current_target]
                .astype("Int64")  # Use nullable integers internally
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

        joined_mapping = joined_mapping.loc[source_index]

        # Apply null strategy at the end
        if dropna:
            joined_mapping = joined_mapping.dropna()

        return joined_mapping

    def get_mapping(
        self,
        source: str,
        target: str,
        source_index: Optional[Union[np.ndarray, pd.Index, pd.Series]] = None,
        validate: Optional[str] = None,
        dropna: bool = False,
    ) -> pd.Series:
        """
        Find mappings from source to target layers using the entire link graph.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.
        source_index :
            Index of the source layer to map from. If None, uses all indices in the
            source layer.
        validate :
            Whether to validate the mapping at each step. If specified, checks if each
            mapping between layers is of the specified type. Options are:
            - "one_to_one" or "1:1": check if join keys are unique in both source and target layers.
            - "one_to_many" or "1:m": check if join keys are unique in the source dataset.
            - "many_to_one" or "m:1": check if join keys are unique in the target dataset.
            - "many_to_many" or "m:m": allowed, but does not result in checks.
        dropna :
            Whether to drop entries with null mappings. If False, returns NaN/pd.NA
            values for missing mappings.

        Returns
        -------
        :
            Series with nodes in the source layer as the index and mapped nodes in
            the target layer as the values. Format depends on `dropna` parameter.

        Notes
        -----
        This function is a convenience wrapper around `get_mapping_paths` that returns
        just the final mapping as a Series. If you need to see the full mapping at
        each step, use `get_mapping_paths`.
        """
        mapping_path = self.get_mapping_paths(
            source, target, source_index, validate=validate, dropna=dropna
        )
        mapping = mapping_path.set_index(source)[target]
        return mapping

    def get_masking(
        self,
        source: str,
        target: str,
        source_index: Optional[Union[np.ndarray, pd.Index, pd.Series]] = None,
    ) -> np.ndarray:
        """Get unique target indices that map to any of the specified source indices.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.
        source_index :
            Source indices to find mappings for. If None, uses all source indices.

        Returns
        -------
        :
            Array of unique target indices that map to the source indices.
        """
        mapping = self.get_mapping(source, target, source_index, dropna=True)
        target_ids = mapping.values
        target_ids = np.unique(target_ids)
        return target_ids

    def get_mapped_nodes(
        self,
        source: str,
        target: str,
        source_index: Optional[Union[np.ndarray, pd.Index, pd.Series]] = None,
        replace_index: bool = True,
        validate: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get features from the target layer for nodes mapped from the source layer.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.
        source_index :
            Source indices to get mappings for. If None, uses all source indices.
        replace_index :
            Whether to replace the target indices with source indices in the result.
        validate :
            Whether to validate the mapping at each step. If specified, checks if each
            mapping between layers is of the specified type. Options are:
            - "one_to_one" or "1:1": check if join keys are unique in both source and target layers.
            - "one_to_many" or "1:m": check if join keys are unique in the source dataset.
            - "many_to_one" or "m:1": check if join keys are unique in the target dataset.
            - "many_to_many" or "m:m": allowed, but does not result in checks.

        Returns
        -------
        :
            DataFrame containing target layer features for the mapped nodes.
        """
        if source_index is None:
            source_index = self.layers[source].nodes_index

        mapping = self.get_mapping(
            source, target, source_index, validate=validate, dropna=True
        )
        target_index = mapping.values
        out = self.layers[target].nodes.reindex(target_index)
        if replace_index:
            out = out.set_index(mapping.index)
        return out

    def assign_from_mapping(
        self,
        source: str,
        target: str,
        columns: Union[str, list, dict],
    ) -> None:
        """Assign values from the target layer to the source layer based on mapping.

        Parameters
        ----------
        source :
            Name of the source layer that will receive the values.
        target :
            Name of the target layer that provides the values.
        columns :
            Column specification:
            - str: Single column name to copy
            - list: List of column names to copy
            - dict: Mapping from source column names to target column names

        Notes
        -----
        This modifies the source layer in place by adding/updating the specified columns.
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

    def apply_mask(self, layer_name: str, mask: Union[np.ndarray, list]) -> "MorphSync":
        """Create a new MorphSync with a masked version of the specified layer.

        Parameters
        ----------
        layer_name :
            Name of the layer to mask.
        mask :
            Boolean mask or indices to apply to the layer.

        Returns
        -------
        :
            New MorphSync with the masked layer and corresponding
            mappings to other layers.
        """
        layer = self.layers[layer_name]
        new_index = layer.nodes.index[mask]
        return self._generate_new_morphology(layer_name, new_index)

    def apply_mask_by_node_index(
        self, layer_name: str, new_index: Union[pd.Index, np.ndarray, list]
    ) -> "MorphSync":
        """Create a new MorphSync with a layer subset by node index.

        Parameters
        ----------
        layer_name :
            Name of the layer to subset.
        new_index :
            Node indices to keep in the new layer.

        Returns
        -------
        :
            New MorphSync container with the subset layer and corresponding
            mappings to other layers.
        """
        return self._generate_new_morphology(layer_name, new_index)

    def _generate_new_morphology(
        self, layer_name: str, new_index: Union[pd.Index, np.ndarray, list]
    ) -> "MorphSync":
        new_morphology = self.__class__(**self.get_params())
        new_morphology._add_layer(
            layer_name,
            self.layers[layer_name].mask_by_node_index(new_index),
        )
        for other_layer_name, other_layer in self.layers.items():
            if other_layer_name != layer_name:
                other_indices = self.get_masking(
                    layer_name, other_layer_name, source_index=new_index
                )
                new_morphology._add_layer(
                    other_layer_name, other_layer.mask_by_node_index(other_indices)
                )
        new_morphology.links = self.links

        return new_morphology

    def get_link_as_layer(self, source: str, target: str) -> Graph:
        """Create a Graph layer representing the mapping between two layers.

        Parameters
        ----------
        source :
            Name of the source layer.
        target :
            Name of the target layer.

        Returns
        -------
        :
            Graph layer where edges connect mapped nodes between source and target layers.
            Source nodes come first in the node list, followed by target nodes.
        """
        mapping = self.get_mapping(source, target, dropna=True)
        source_index = mapping.index
        target_index = mapping.values
        source_nodes = self.layers[source].nodes.loc[source_index]
        target_nodes = self.layers[target].nodes.loc[target_index]
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

    def query_nodes(self, layer_name: str, query_str: str) -> "MorphSync":
        """Create a new MorphSync by querying nodes in a specific layer.

        Parameters
        ----------
        layer_name :
            Name of the layer to query.
        query_str :
            Query string to pass to pandas DataFrame.query() method.

        Returns
        -------
        :
            New MorphSync container with the queried layer and corresponding
            mappings to other layers.
        """
        layer_query = self.layers[layer_name].query_nodes(query_str)
        new_index = layer_query.nodes.index
        return self._generate_new_morphology(layer_name, new_index)
