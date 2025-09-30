import pandas as pd

from .base import Layer


class Table(Layer):
    def __init__(self, table: pd.DataFrame, *args, **kwargs):
        """Initialize a Table layer.

        Parameters
        ----------
        table :
            DataFrame containing tabular data, or tuple containing the DataFrame.
        *args : tuple
            Additional arguments passed to the parent Layer class.
        **kwargs : dict
            Additional keyword arguments passed to the parent Layer class.
        """
        if isinstance(table, tuple):
            table = table[0]
        super().__init__(table, None, *args, **kwargs)

    def __repr__(self) -> str:
        """Return a string representation of the Table."""
        return f"Table(rows={len(self.nodes)})"

    @property
    def table(self) -> pd.DataFrame:
        """Access the table data as a DataFrame. Alias for nodes."""
        return self.nodes
