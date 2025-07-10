import numpy as np
import pandas as pd

from .base import FacetFrame


class Points(FacetFrame):
    def __init__(self, points, *args, **kwargs):
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

    def __repr__(self):
        return f"Points(points={self.points.shape})"

    @property
    def index(self):
        return self.nodes.index
