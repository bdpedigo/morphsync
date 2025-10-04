import pooch

from . import __version__ as version

__all__ = ["fetch_minnie_sample"]

minnie_data = pooch.create(
    path=pooch.os_cache("morphsync"),
    base_url="https://github.com/bdpedigo/morphsync/raw/{version}/data/minnie65_864691134918461194/",
    version=version,
    registry={
        "layers/level2_nodes_nodes.csv.gz": "sha256:8a2d189db2c9e5789b672cee88fac4d6a50d6e68e85f988d6a00ea0cfd42bda7",
        "layers/level2_skeleton_facets.csv.gz": "sha256:186878af524777f4aec8d95fec1f6ca8af9825d30252304e7baf9fa4c83d6860",
        "layers/level2_skeleton_nodes.csv.gz": "sha256:2008ec2dfee3a0cdad5fa33ed3acd718f72a315a19e3f9a168e96fcf5cb6dffb",
        "layers/mesh_facets.csv.gz": "sha256:54baf257a697a59e59de63f7a5df0593ef136ec5377e8daa664f976cdb4c7234",
        "layers/mesh_nodes.csv.gz": "sha256:b7ddd315ffe39f33f54cf8abb55c03fb321e4d1b0c28939567ebcc28526d8584",
        "layers/post_synapses_nodes.csv.gz": "sha256:5ddc9aeea616c1e645ad1e1c78e9989e661cfb33f36133a498d660b85dedaf1e",
        "layers/pre_synapses_nodes.csv.gz": "sha256:4cddac37d6ed5327b7664ed00838417754265b796f4c62a5b32ca57aed3e1760",
        "layers/segclr_nodes.csv.gz": "sha256:cbc146eb468fc7aa42d313f5c074485abf407838da47000377ff38a5734932bd",
        "links/level2_nodes_to_level2_skeleton_mapping.csv.gz": "sha256:4cbd3062f56e26fd95ba9c134fbdf31d046faf7a7d00d8f2c87c16b940df7216",
        "links/level2_nodes_to_mesh_mapping.csv.gz": "sha256:a380cf17499e4ea1c8cec932a7fc02241a3408bf0998a7ebb6711b80d0117f00",
        "links/level2_nodes_to_post_synapses_mapping.csv.gz": "sha256:19e36470c83a0d4b9ba701869214f31b3a74e3641a34c00839bf9acf6c5a7949",
        "links/level2_nodes_to_pre_synapses_mapping.csv.gz": "sha256:e0c26c9d5f49b59b4f25af2b4864cc45617c49b1e73165450e51c1d4f5f29641",
        "links/level2_nodes_to_segclr_mapping.csv.gz": "sha256:fba4d82b93b9f5d4c9dfcd9d9ae831e1bcc98e6639a0eb5fd285106f6e9488c6",
        "links/mesh_to_post_synapses_mapping.csv.gz": "sha256:a0e16b760565b74b1e82660dcc6558b400fa2a3c92cdc445b2995a72619b5c13",
        "links/mesh_to_pre_synapses_mapping.csv.gz": "sha256:5f58d0f2ec5b22c47143516cb9f453e6a19b6d92a2201b312cd16b193863d887",
        "morph_info.json": "sha256:6b66f2d07236eef619d6b1550aa2ea1ef180a75aa7122b85bebe30ca08e5801e",
    },
)


def fetch_minnie_sample():
    """Fetch the Minnie sample neuron.

    Returns
    -------
    str
        Path to the downloaded dataset.
    """
    # return minnie_data.fetch("minnie_mouse.zip", processor=pooch.Unzip())
    path = minnie_data.fetch("morph_info.json")
    print(path)
