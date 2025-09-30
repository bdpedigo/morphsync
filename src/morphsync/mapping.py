import pandas as pd
from sklearn.neighbors import NearestNeighbors


def project_points_to_nearest(
    source_points, target_points, distance_threshold=None, return_distances=False
):
    """Project source points to their nearest neighbors in target points.

    Parameters
    ----------
    source_points : pd.DataFrame or np.ndarray
        Array or DataFrame of shape (n_source, 3) containing source point coordinates.
    target_points : pd.DataFrame or np.ndarray
        Array or DataFrame of shape (n_target, 3) containing target point coordinates.
    distance_threshold : float, optional
        Maximum distance threshold. Points farther than this distance will be
        mapped to -1. By default None (no threshold).
    return_distances : bool, default False
        If True, return both indices and distances. If False, return only indices.

    Returns
    -------
    indices : np.ndarray
        Array of shape (n_source,) containing indices of nearest target points.
        Values of -1 indicate no mapping within distance threshold.
    distances : np.ndarray, optional
        Array of shape (n_source,) containing distances to nearest neighbors.
        Only returned if return_distances=True.
    """
    if isinstance(source_points, pd.DataFrame):
        source_points = source_points.values
    if isinstance(target_points, pd.DataFrame):
        target_points = target_points.values
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(target_points)

    distances, indices = nn.kneighbors(source_points)
    indices = indices.reshape(-1)
    distances = distances.reshape(-1)
    if distance_threshold is not None:
        indices[distances > distance_threshold] = -1

    if return_distances:
        return indices, distances
    else:
        return indices
