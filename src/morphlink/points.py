import numpy as np
import pandas as pd

from .base import FacetFrame
from .integration import import_pyvista


class Points(FacetFrame):
    def __init__(self, points):
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
        super().__init__(points, None)

    def __repr__(self):
        return f"Points(points={self.points.shape})"

    def to_pyvista(self):
        pv = import_pyvista()
        return pv.PolyData(self.points)
