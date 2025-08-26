"""
Utilities for loading graph datasets.

The module provides a DataLoader class that can load graph datasets torch-geometric.data.Dataset
and return a polars DataFrame of the edges and nodes. It contains methods to convert the graph
into a raphtory graph and a networkx graph.
"""

import logging
from pathlib import Path
import polars as pl
from torch_geometric.data import Data, InMemoryDataset
from torch import Tensor
from typing import Optional, Callable

from .utils import (
    polars_to_tg,
    polars_to_raphtory,
    polars_to_networkx,
    polars_to_temporal_data,
)


class BaseDataset(InMemoryDataset):
    """
    Wrapper for a PyTorch Geometric Dataset.
    """

    def __init__(
        self,
        root: str | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """
        Initialize a new BaseDataset instance.

        Args:
            root (str or Path): The root directory where the dataset is stored.
            transform (callable, optional): A function to apply transformations to the data.
            pre_transform (callable, optional): A function to apply preprocessing transformations before the main transform.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(root, transform, pre_transform)

    @property
    def raw_dir(self) -> str:
        return str(Path(self.root) / "raw")

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root) / "processed")

    @property
    def processed_file_names(self) -> str | list[str] | tuple[str, ...]:
        """
        The processed file names. Override in subclasses.
        """
        return ["edge_data.parquet", "node_data.parquet"]

    @property
    def raw_file_names(self) -> list[str]:
        """
        The raw file names. Override in subclasses.
        """
        return []

    def _load_polars(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Load the processed edge and node Polars DataFrames.
        """
        if hasattr(self, "edge_df") and hasattr(self, "node_df"):
            print("Loading edge and node data from memory")
            return self.edge_df, self.node_df
        processed_dir = Path(self.processed_dir)
        if not (edge_data := Path(processed_dir / "edge_data.parquet")).exists():
            raise FileNotFoundError(f"Parquet file {edge_data} not found.")
        edge_df = pl.read_parquet(processed_dir / "edge_data.parquet")
        if Path(processed_dir / "node_data.parquet").exists():
            node_df = pl.read_parquet(processed_dir / "node_data.parquet")
        else:
            node_df = None
        return edge_df, node_df

    def _to_raphtory(self):
        """
        Convert the processed edge and node Polars DataFrames to a Raphtory graph.
        """
        edge_df, node_df = self._load_polars()
        graph = polars_to_raphtory(edge_df, node_df)
        return graph

    def _to_torch_geometric(self) -> tuple[Data, Optional[dict[str, Tensor]]]:
        """
        Convert the processed edge and node Polars DataFrames to a PyTorch Geometric dataset.
        """
        edge_df, node_df = self._load_polars()
        data, slices = self.collate(polars_to_tg(edge_df, node_df, self.pre_transform))
        return data, slices

    def _to_networkx(self):
        """
        Convert the processed edge and node Polars DataFrames to a NetworkX graph.
        """
        edge_df, node_df = self._load_polars()
        return polars_to_networkx(edge_df, node_df)

    def _to_temporal_data(self):
        """
        Convert the processed edge and node Polars DataFrames to a TemporalData object.
        """
        edge_df, node_df = self._load_polars()
        return polars_to_temporal_data(edge_df, node_df)

    def to(self, fmt: str):  # pylint: disable=arguments-renamed
        """
        Convert the dataset to a different format.

        Args:
            fmt: Target format ('raphtory', 'polars', 'torch-geometric', 'networkx', 'temporal-data', or device string for torch tensors)

        Returns:
            Dataset in the requested format
        """
        match fmt:
            case "raphtory":
                return self._to_raphtory()
            case "polars":
                return self.edge_df, self.node_df
            case "torch-geometric":
                if not hasattr(self, "data"):
                    self.data, _ = self._to_torch_geometric()
                return self.data
            case "networkx":
                return self._to_networkx()
            case "temporal-data":
                return self._to_temporal_data()
            case _:
                return super().to(fmt)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
