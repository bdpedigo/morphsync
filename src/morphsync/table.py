from .base import FacetFrame


class Table(FacetFrame):
    def __init__(self, dataframe, *args, **kwargs):
        if isinstance(dataframe, tuple):
            dataframe = dataframe[0]
        super().__init__(dataframe, None, *args, **kwargs)

    def __repr__(self):
        return f"Table(rows={len(self.nodes)})"

    @property
    def table(self):
        return self.nodes
