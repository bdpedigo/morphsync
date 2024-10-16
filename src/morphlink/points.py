import numpy as np
import pandas as pd

from .base import FacetFrame
from .integration import import_pyvista


class Points(FacetFrame):
    def __init__(self, points, spatial_columns=None):
        if spatial_columns is None:
            spatial_columns = ["x", "y", "z"]
        if isinstance(points, np.ndarray):
            if points.shape == (3,):
                points = points.reshape(1, 3)
            points = pd.DataFrame(points, columns=spatial_columns)
        super().__init__(points, None, spatial_columns=spatial_columns)

    def __repr__(self):
        return f"Points(points={self.points.shape})"

    def to_pyvista(self):
        pv = import_pyvista()
        return pv.PolyData(self.points)
