from torch_geometric.datasets import Planetoid
from pathlib import Path
from .registry import register_dataset
from .base import BaseDataset
from .utils import tg_to_polars


@register_dataset("PubMed")
class PubMedDataset(BaseDataset):
    """
    PubMed dataset from the Planetoid dataset collection.

    The PubMed dataset is a citation network of 19,717 scientific publications
    from the PubMed database classified into 3 classes based on diabetes research.
    Each node has a 500-dimensional feature vector derived from the TF-IDF
    weighted word vectors of the paper abstracts.

    Dataset Statistics:
    - Nodes: 19,717
    - Edges: 88,648
    - Features: 500
    - Classes: 3 (Diabetes Mellitus Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2)
    - Type: Citation network

    This is the largest of the Planetoid datasets and provides a good test case
    for scalability and performance of graph neural network methods.

    References:
        "Collective Classification in Network Data"
        Prithviraj Sen, Galileo Namata, Mustafa Bilgic, Lise Getoor, Brian Gallagher, Tina Eliassi-Rad
        AI Magazine, 2008

        "Revisiting Semi-Supervised Learning with Graph Embeddings"
        Zhilin Yang, William Cohen, Ruslan Salakhudinov
        ICML 2016
    """

    def __init__(self, root: str | None = None, **kwargs):
        """
        Initialize the PubMed dataset.

        Args:
            root: Root directory to store the dataset (default: data/)
            **kwargs: Additional arguments passed to Planetoid dataset
        """
        # Logger is set up in BaseDataset
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data")
        super().__init__(root, **kwargs)
        kwargs.setdefault("name", "PubMed")
        data = Planetoid(root, **kwargs)
        self.data = data[0]
        self.edge_df, self.node_df = tg_to_polars(data)
