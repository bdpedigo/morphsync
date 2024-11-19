import pandas as pd
from sklearn.neighbors import NearestNeighbors


def project_points_to_nearest(
    source_points, target_points, distance_threshold=None, return_distances=False
):
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
