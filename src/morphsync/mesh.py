from .base import FacetFrame


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

    def __repr__(self):
        return f"Mesh(vertices={self.n_vertices}, faces={self.n_facets})"

    @classmethod
    def from_dict(cls, data):
        return cls(data["vertices"], data["faces"])
