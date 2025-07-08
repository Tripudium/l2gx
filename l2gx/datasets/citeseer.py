from torch_geometric.datasets import Planetoid
from pathlib import Path
from .registry import register_dataset
from .base import BaseDataset
from .utils import tg_to_polars


@register_dataset("CiteSeer")
class CiteSeerDataset(BaseDataset):
    """
    CiteSeer dataset from the Planetoid dataset collection.
    
    The CiteSeer dataset is a citation network of 3,327 scientific publications
    classified into 6 classes. Each node has a 3,703-dimensional feature vector
    indicating word presence/absence. This is a classic benchmark dataset for
    node classification tasks.
    
    Dataset Statistics:
    - Nodes: 3,327
    - Edges: 9,104  
    - Features: 3,703
    - Classes: 6 (Agents, AI, DB, IR, ML, HCI)
    - Type: Citation network
    
    References:
        "Collective Classification in Network Data"
        Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, Tina Eliassi-Rad
        AI Magazine, 2008
    """

    def __init__(self, root: str | None = None, **kwargs):
        """
        Initialize the CiteSeer dataset.
        
        Args:
            root: Root directory to store the dataset (default: data/)
            **kwargs: Additional arguments passed to Planetoid dataset
        """
        # Logger is set up in BaseDataset
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data")
        super().__init__(root, **kwargs)
        kwargs.setdefault("name", "CiteSeer")
        data = Planetoid(root, **kwargs)
        self.data = data[0]
        self.edge_df, self.node_df = tg_to_polars(data)