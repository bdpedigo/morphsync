# MorphSync

MorphSync is a Python library for working with multi-layered morphological data structures. It provides a unified framework for managing and synchronizing different representations of morphological data such as meshes, point clouds, graphs, and tabular data.

## Key Features

- **Multi-layer data management**: Handle meshes, point clouds, graphs, and tables in a unified framework
- **Automatic mapping**: Create and manage mappings between different data layers
- **Spatial operations**: Built-in support for spatial queries and nearest neighbor operations
- **Flexible data structures**: Work with pandas DataFrames and numpy arrays seamlessly
- **Graph-based linking**: Use graph algorithms to find mappings across multiple layers

## Quick Start

```python
from morphsync import MorphSync

# Create a new morphology container
morphology = MorphSync(name=12345678)

# Add different layers
morphology.add_mesh("mesh", mesh_data)
morphology.add_points("synapses", point_data)
morphology.add_graph("skeleton", graph_data)

# Create mappings between layers
morphology.add_link("synapses", "mesh", mapping=synapse_to_mesh_mapping)

# Query mappings
mapping = morphology.get_mapping("synapses", "mesh")
```

## Core Concepts

### Layers

MorphSync organizes data into **layers**, where each layer represents a different aspect of morphological data:

- **Mesh**: 3D surface representations with vertices and faces
- **Points**: Point clouds such as annotations
- **Graph**: Network structures like skeletons or connectivity graphs
- **Table**: Tabular data without a spatial component

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
