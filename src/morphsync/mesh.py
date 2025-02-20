from .base import FacetFrame
from .integration import import_pyvista


class Mesh(FacetFrame):
    def __init__(self, input, **kwargs):
        if hasattr(input, "vertices") and hasattr(input, "faces"):
            super().__init__(input.vertices, input.faces, **kwargs)
        elif isinstance(input, tuple):
            super().__init__(input[0], input[1], **kwargs)
        else:
            raise NotImplementedError(
                "Only accepts objects with 'vertices' and 'faces' attributes"
            )

    @property
    def faces(self):
        return self.facets_positional

    def __str__(self):
        return f"Mesh(vertices={self.vertices.shape}, faces={self.faces.shape})"

    def __repr__(self):
        return f"Mesh(vertices={self.vertices.shape}, faces={self.faces.shape})"

    def to_pyvista(self):
        pv = import_pyvista()

        return pv.make_tri_mesh(self.vertices, self.faces)

    @classmethod
    def from_dict(cls, data):
        return cls(data["vertices"], data["faces"])
