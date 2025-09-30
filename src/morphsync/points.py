from typing import Union

import numpy as np
import pandas as pd

from .base import Layer


class Points(Layer):
    def __init__(self, points: Union[np.ndarray, pd.DataFrame, tuple], *args, **kwargs):
        """Initialize a Points layer.

        Parameters
        ----------
        points :
            Point coordinates as an array, DataFrame, or tuple containing points.
            If array, should be shape (n_points, 3) or (3,) for a single point.
        *args : tuple
            Additional arguments passed to the parent Layer class.
        **kwargs : dict
            Additional keyword arguments passed to the parent Layer class.
        """
        if isinstance(points, tuple):
            # TODO possibly dumb hack for compatibility with mask_nodes etc.
            # but currently, all FacetFrames are expected to take 2 arguments
            # (points + facets)
            points = points[0]
        if isinstance(points, np.ndarray):
            if points.shape == (3,):
                points = points.reshape(1, 3)
            points = pd.DataFrame(points)
        elif isinstance(points, pd.DataFrame):
            pass
        kwargs["relation_columns"] = []
        super().__init__(points, None, *args, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the Points."""
        return f"Points(points={self.points.shape})"

    @property
    def index(self) -> pd.Index:
        """Index of the points. Alias for nodes.index."""
        return self.nodes.index

    @property
    def is_spatially_valid(self) -> bool:
        """Check if the points have valid spatial structure.

        Returns
        -------
        :
            True if points are 3D and non-empty.
        """
        return (self.points.shape[1] == 3) & (self.points.shape[0] > 0)
