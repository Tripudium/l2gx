import shutil
from torch_geometric.datasets import DGraphFin
from pathlib import Path

from .registry import register_dataset
from .base import BaseDataset
from .utils import tg_to_polars


@register_dataset("DGraph")
class DGraphDataset(BaseDataset):
    """
    DGraph dataset
    https://github.com/DGraphXinye/DGraphFin_baseline
    """

    def __init__(
        self, root: str | None = None, source_file: str | None = None, **kwargs
    ):
        """
        Initialize the DGraph dataset.
        Dataset needs to be downloaded from the website and placed in the data folder.
        Give path to source file as source_file argument.
        """
        # Logger is set up in BaseDataset
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data" / "dgraph")
        self.source_file = source_file
        self._raw_paths = [source_file]
        super().__init__(root, **kwargs)
        data = DGraphFin(root)
        self.data = data[0]
        self.edge_df, self.node_df = tg_to_polars(data)
        self.raphtory_graph = self._to_raphtory()

    @property
    def raw_paths(self):
        return self._raw_paths

    @raw_paths.setter
    def raw_paths(self, value):
        self._raw_paths = value

    def download(self):
        """
        Download the DGraph dataset.
        """
        if self.source_file is not None:
            raw_dir = Path(self.raw_dir)
            source_file = Path(self.source_file)
            target_file = raw_dir / source_file.name
            if target_file.exists():
                self.logger.info(
                    f"File {target_file} already exists in raw directory, skipping download"
                )
                return
            if not source_file.exists():
                raise FileNotFoundError(
                    f"Raw file path {self.source_file} does not exist."
                )
            if not raw_dir.exists():
                raw_dir.mkdir(parents=True, exist_ok=True)
            if source_file.is_file():
                shutil.copy2(source_file, raw_dir)
                self.logger.info(f"Copied file from {source_file} to {raw_dir}")
            else:
                raise FileNotFoundError(
                    f"Raw file path {self.source_file} is not a file."
                )
