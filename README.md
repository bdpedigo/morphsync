# MorphSync

MorphSync is a Python library for working with multi-layered morphological data.
It provides a unified framework for managing and synchronizing different representations
of morphological data such as meshes, point clouds, graphs, and tabular data.

## Key Features

- **Multi-layer data management**: Handle meshes, point clouds, graphs, and tables in a unified framework
- **Linking**: Create and manage mappings between different data layers
- **Link-based operations**: Apply masks/selections or features from one data layer to another
- **Transitivity**: Easily apply transitive mappings across multiple layers
- **Lightweight**: Intentionally light on algorithms, use your own tools but stop worrying about mappings

## Quick Start

```python
from morphsync import MorphSync

# Create a new morphology container
morphology = MorphSync(name=12345678)

mesh_data # object with vertices and faces
point_data # dataframe of points with optional features
graph_data # object with nodes and edges, either with optional features

# Add different layers
morphology.add_mesh("mesh", mesh_data)
morphology.add_points("synapses",
    point_data, 
    spatial_columns=["x", "y", "z"],
)
morphology.add_graph("skeleton",
    graph_data,
    spatial_columns=["x", "y", "z"],
    relation_columns=["source", "target"],
)

synapse_to_mesh_mapping # series mapping synapse indices to mesh indices
mesh_to_skeleton_mapping # series mapping mesh indices to skeleton indices

# Create mappings between layers
morphology.add_link("synapses", "mesh", mapping=synapse_to_mesh_mapping)
morphology.add_link("mesh", "skeleton", mapping=mesh_to_skeleton_mapping)

# Select all aspects of the morphology that map to the skeleton's axon labeling
# Note that transitive mapping (synapses <-> mesh <-> skeleton) works here!
mapping = morphology.query_nodes("skeleton", "compartment == 'axon'")

# Add a column to synapses which indicates the local skeleton radius
morphology.assign_from_mapping("synapses", "skeleton", ["radius"])
```

## Core Concepts

### Layers

MorphSync organizes data into **layers**, where each layer represents a different aspect of morphological data:

- **Table**: Tabular data without a spatial component
- **Points**: Point clouds such as annotations
- **Graph**: Network structures like skeletons or connectivity graphs
- **Mesh**: 3D surface representations with vertices and faces

Note that it's possible to have multiple layers of the same type (e.g., multiple skeleton layers).

### Links

**Links** define relationships and mappings between layers. They enable you to:

- Use one layer to query or filter another layer
- Find features from one layer based on mappings; for instance, finding the radius of skeleton nodes that synapses map to
- Perform transitive mappings across multiple layers

## Installation

```bash
pip install morphsync
```

or, to add to a `uv` managed project or environment:

```bash
uv add morphsync
```
