import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .morph import MorphSync


def load_morphsync(path: Union[Path, str]) -> MorphSync:
    """Load a MorphSync object from a directory.

    THIS FUNCTION IS FOR TESTING AND THE API/FILE STORAGE IS LIKELY TO CHANGE.

    Parameters
    ----------
    path :
        Path to the directory containing the MorphSync data.

    Returns
    -------
    :
        The loaded MorphSync object.
    """
    read_info = json.load(open(path / "morph_info.json"))
    loaded_morph = MorphSync(name=read_info["name"])
    for layer_name, layer_info in read_info["layers"].items():
        layer_path = path / f"layers/{layer_name}_nodes.csv.gz"
        if Path.exists(layer_path):
            nodes = pd.read_csv(layer_path, index_col=0)
        else:
            nodes = None
        layer_path = path / f"layers/{layer_name}_facets.csv.gz"
        if Path.exists(layer_path):
            facets = pd.read_csv(layer_path, index_col=0)
            if np.array_equal(facets.columns, ["0", "1", "2"]):
                facets.columns = facets.columns.astype(int)
        else:
            facets = None
        loaded_morph.add_layer(
            name=layer_name,
            data=(nodes, facets),
            layer_type=layer_info["layer_type"],
            spatial_columns=layer_info.get("spatial_columns", []),
            relation_columns=layer_info.get("relation_columns", []),
        )

    for link_info in read_info["links"]:
        link_path = (
            path
            / f"links/{link_info['source']}_to_{link_info['target']}_mapping.csv.gz"
        )
        if Path.exists(link_path):
            mapping = pd.read_csv(link_path)
            loaded_morph.add_link(
                link_info["source"],
                link_info["target"],
                mapping=mapping,
                reciprocal=True,
            )
        else:
            continue

    return loaded_morph
