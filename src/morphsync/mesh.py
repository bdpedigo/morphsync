from .base import FacetFrame


class Mesh(FacetFrame):
    def __init__(self, input, **kwargs):
        if hasattr(input, "vertices") and hasattr(input, "faces"):
            vertices = input.vertices
            faces = input.faces
        elif isinstance(input, tuple):
            vertices, faces = input
        else:
            raise NotImplementedError(
                "Only accepts objects with 'vertices' and 'faces' attributes"
            )
        super().__init__(vertices, faces, **kwargs)

    @property
    def faces(self):
        return self.facets_positional

    def __repr__(self):
        return f"Mesh(vertices={self.n_vertices}, faces={self.n_facets})"

    @classmethod
    def from_dict(cls, data):
        return cls(data["vertices"], data["faces"])

    @property
    def is_spatially_valid(self):
        is_valid = (self.vertices.shape[1] == 3) & (self.vertices.shape[0] > 0)
        is_valid &= (self.faces.shape[1] == 3) & (self.faces.shape[0] > 0)
        return is_valid

    @property
    def mesh(self):
        return (self.vertices, self.faces)