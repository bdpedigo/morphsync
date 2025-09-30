from .base import Layer


class Mesh(Layer):
    def __init__(self, input, **kwargs):
        """Initialize a Mesh layer.

        Parameters
        ----------
        input : object or tuple
            Either an object with 'vertices' and 'faces' attributes, or a tuple
            of (vertices, faces).
        **kwargs : dict
            Additional keyword arguments passed to the parent Layer class.
        
        Raises
        ------
        NotImplementedError
            If input doesn't have the required attributes or format.
        """
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
        """Faces as a numpy array in positional indexing."""
        return self.facets_positional

    def __repr__(self):
        """Return a string representation of the Mesh."""
        return f"Mesh(vertices={self.nodes.shape}, faces={self.faces.shape})"

    @classmethod
    def from_dict(cls, data):
        """Create a Mesh from a dictionary containing vertices and faces.
        
        Parameters
        ----------
        data : dict
            Dictionary with 'vertices' and 'faces' keys.
            
        Returns
        -------
        Mesh
            A new Mesh instance.
        """
        return cls(data["vertices"], data["faces"])

    @property
    def is_spatially_valid(self):
        """Check if the mesh has valid spatial structure.
        
        Returns
        -------
        bool
            True if vertices are 3D, faces are triangular, and both are non-empty.
        """
        is_valid = (self.vertices.shape[1] == 3) & (self.vertices.shape[0] > 0)
        is_valid &= (self.faces.shape[1] == 3) & (self.faces.shape[0] > 0)
        return is_valid

    @property
    def mesh(self):
        """Get the mesh as a tuple of (vertices, faces)."""
        return (self.vertices, self.faces)
