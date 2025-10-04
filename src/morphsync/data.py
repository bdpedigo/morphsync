import pooch

from . import __version__ as version

data = pooch.create(
    path=pooch.os_cache("morphsync"),
    base_url="https://github.com/bdpedigo/morphsync/raw/{version}/data/",
    version=version,
)


def fetch_minnie_sample():
    """Fetch the Minnie sample neuron.

    Returns
    -------
    str
        Path to the downloaded dataset.
    """
    return data.fetch("minnie_mouse.zip", processor=pooch.Unzip())
