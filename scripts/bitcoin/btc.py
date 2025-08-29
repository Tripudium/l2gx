"""
Bitcoin Transaction Graph Dataset

This dataset contains Bitcoin transaction data with node features and transaction edges.
Data is loaded from parquet files containing millions of Bitcoin addresses and their transactions.

The dataset includes:
- Node features: Various transaction statistics and cluster information for Bitcoin addresses
- Transaction edges: Aggregated transaction relationships between addresses  
- Labels: Entity types (INDIVIDUAL, EXCHANGE, MIXER, etc.) for some addresses

Data is processed using Polars for efficient handling of large datasets.
"""

import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from .base import BaseDataset
from .registry import register_dataset


@register_dataset("btc")
@register_dataset("BTC") 
@register_dataset("Bitcoin")
class BTCDataset(BaseDataset):
    """
    Bitcoin Transaction Graph Dataset.
    
    This dataset contains Bitcoin transaction network data with node features
    and labels for different entity types.
    
    Args:
        root: Root directory where the dataset is stored
        subset: Optional subset specification (e.g., 'small' for testing)
        max_nodes: Maximum number of nodes to include (None for all)
        seed: Random seed for reproducible subsampling
        
    Attributes:
        num_classes: Number of label classes (13: null + 12 entity types)
        label_names: List of label names in order
    """
    
    # Label mapping: null=0, then alphabetical order of non-null labels
    LABEL_NAMES = [
        "UNLABELED",     # 0 - for null labels
        "BET",           # 1
        "BRIDGE",        # 2  
        "EXCHANGE",      # 3
        "FAUCET",        # 4
        "INDIVIDUAL",    # 5
        "MARKETPLACE",   # 6
        "MINING",        # 7
        "MIXER",         # 8
        "PONZI",         # 9
        "RANSOMWARE",    # 10
        "OTHER"          # 11 - for any other labels
    ]
    
    def __init__(
        self,
        root: Optional[str] = None,
        subset: Optional[str] = None,
        max_nodes: Optional[int] = None,
        seed: int = 42,
    ):
        self.subset = subset
        self.max_nodes = max_nodes  
        self.seed = seed
        
        # Set default root to data/BTCGraph
        if root is None:
            # Assume we're in l2gx/datasets, so go up to project root
            project_root = Path(__file__).parent.parent.parent
            root = str(project_root / "data" / "BTCGraph")
        
        super().__init__(root=root)
        
        # Validate data files exist
        self.nodes_file = Path(self.root) / "node_features.parquet"
        self.edges_file = Path(self.root) / "transaction_edges.parquet"
        
        if not self.nodes_file.exists():
            raise FileNotFoundError(f"Node features file not found: {self.nodes_file}")
        if not self.edges_file.exists():
            raise FileNotFoundError(f"Transaction edges file not found: {self.edges_file}")
    
    @property
    def num_classes(self) -> int:
        """Number of label classes."""
        return len(self.LABEL_NAMES)
    
    @property
    def label_names(self) -> list[str]:
        """List of label names in order."""
        return self.LABEL_NAMES.copy()
    
    def _encode_label(self, label_str: Optional[str]) -> int:
        """Convert string label to integer."""
        if label_str is None or label_str == "null":
            return 0  # UNLABELED
        
        try:
            return self.LABEL_NAMES.index(label_str)
        except ValueError:
            # Unknown label -> OTHER
            return len(self.LABEL_NAMES) - 1
    
    def _load_nodes(self) -> tuple[pl.DataFrame, dict[int, int]]:
        """Load and process node features."""
        print(f"Loading nodes from {self.nodes_file}")
        
        # Select relevant columns first
        nodes_lazy = pl.scan_parquet(self.nodes_file).select([
            'alias',
            'degree', 'degree_in', 'degree_out',
            'total_transactions_in', 'total_transactions_out', 
            'total_sent', 'total_received',
            'cluster_size',
            'label'
        ])
        
        # Apply subsampling if requested
        if self.max_nodes is not None:
            print(f"Sampling {self.max_nodes} nodes...")
            # Collect first, then sample (for large datasets, consider streaming)
            nodes = nodes_lazy.collect().sample(n=self.max_nodes, seed=self.seed)
        else:
            nodes = nodes_lazy.collect()
        
        print(f"Loaded {len(nodes)} nodes")
        
        # Create node ID mapping (alias -> consecutive IDs starting from 0)
        unique_aliases = nodes['alias'].unique().sort()
        alias_to_id = {alias: i for i, alias in enumerate(unique_aliases)}
        
        return nodes, alias_to_id
    
    def _load_edges(self, alias_to_id: dict[int, int]) -> pl.DataFrame:
        """Load and process transaction edges."""
        print(f"Loading edges from {self.edges_file}")
        
        # Get set of valid node aliases
        valid_aliases = set(alias_to_id.keys())
        
        # Load edges and filter to only include nodes we have
        edges_lazy = pl.scan_parquet(self.edges_file)
        
        # Select relevant columns and filter for valid nodes
        edges = edges_lazy.select([
            'a', 'b', 'total', 'total_sent'
        ]).filter(
            pl.col('a').is_in(list(valid_aliases)) &
            pl.col('b').is_in(list(valid_aliases))
        ).collect()
        
        print(f"Loaded {len(edges)} edges between selected nodes")
        
        return edges
    
    def _create_node_features(self, nodes: pl.DataFrame) -> torch.Tensor:
        """Create node feature matrix."""
        # Select numerical features for the feature matrix
        feature_cols = [
            'degree', 'degree_in', 'degree_out',
            'total_transactions_in', 'total_transactions_out',
            'total_sent', 'total_received', 
            'cluster_size'
        ]
        
        # Extract features and handle nulls
        features = nodes.select(feature_cols).fill_null(0)
        feature_array = features.to_numpy().astype(np.float32)
        
        # Log transform large values to normalize
        feature_array = np.log1p(np.maximum(feature_array, 0))
        
        return torch.from_numpy(feature_array)
    
    def _create_labels(self, nodes: pl.DataFrame) -> torch.Tensor:
        """Create label tensor."""
        # Convert string labels to integers
        labels = []
        for label_str in nodes['label'].to_list():
            labels.append(self._encode_label(label_str))
        
        return torch.tensor(labels, dtype=torch.long)
    
    def _create_edge_index(self, edges: pl.DataFrame, alias_to_id: dict[int, int]) -> torch.Tensor:
        """Create edge index tensor."""
        if len(edges) == 0:
            # No edges - return empty tensor with correct shape
            return torch.empty((2, 0), dtype=torch.long)
        
        # Map aliases to consecutive node IDs
        source_ids = [alias_to_id[alias] for alias in edges['a'].to_list()]
        target_ids = [alias_to_id[alias] for alias in edges['b'].to_list()]
        
        # Create bidirectional edges (undirected graph)
        edge_list = []
        edge_list.extend(zip(source_ids, target_ids))
        edge_list.extend(zip(target_ids, source_ids))  # Reverse direction
        
        # Convert to tensor
        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        
        edge_array = np.array(edge_list, dtype=np.int64).T
        return torch.from_numpy(edge_array)
    
    def _create_edge_attr(self, edges: pl.DataFrame) -> torch.Tensor:
        """Create edge attribute tensor."""
        if len(edges) == 0:
            # No edges - return empty tensor with correct shape  
            return torch.empty((0, 2), dtype=torch.float32)
        
        # Use total transactions and total sent as edge features  
        edge_features = edges.select(['total', 'total_sent']).fill_null(0).to_numpy()
        
        # Log transform and normalize
        edge_features = np.log1p(np.maximum(edge_features, 0)).astype(np.float32)
        
        # Create bidirectional edge attributes
        edge_attr = np.vstack([edge_features, edge_features])  # Duplicate for bidirectional
        
        return torch.from_numpy(edge_attr)
    
    def get(self, idx: int = 0) -> Data:
        """Get the Bitcoin transaction graph data."""
        if idx != 0:
            raise IndexError(f"BTCDataset only has one graph (idx=0), got {idx}")
        
        print("Loading Bitcoin transaction graph...")
        
        # Load nodes and create mapping
        nodes, alias_to_id = self._load_nodes()
        
        # Load edges
        edges = self._load_edges(alias_to_id)
        
        # Create PyTorch Geometric data
        print("Creating PyTorch Geometric data...")
        
        x = self._create_node_features(nodes)
        y = self._create_labels(nodes)  
        edge_index = self._create_edge_index(edges, alias_to_id)
        
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            num_nodes=len(nodes)
        )
        
        # Add metadata
        data.num_classes = self.num_classes
        data.label_names = self.label_names
        
        print(f"Created graph with {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
        print(f"Node features: {data.x.shape}")
        print(f"Labels: {data.y.shape}, {self.num_classes} classes")
        
        # Print label distribution
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        print("\nLabel distribution:")
        for label_id, count in zip(unique_labels.tolist(), counts.tolist()):
            label_name = self.label_names[label_id]
            print(f"  {label_name}: {count:,} ({count/len(data.y)*100:.1f}%)")
        
        return data
    
    def len(self) -> int:
        """Dataset size (always 1 graph)."""
        return 1
    
    def __repr__(self) -> str:
        return (f"BTCDataset("
                f"root={self.root}, "
                f"subset={self.subset}, "
                f"max_nodes={self.max_nodes})")


@register_dataset("btc-reduced")
@register_dataset("btc_reduced")
@register_dataset("BTC-Reduced")
class BTCReducedDataset(BTCDataset):
    """
    Bitcoin Transaction Graph Dataset - Labeled Nodes Only.
    
    This dataset contains only labeled Bitcoin nodes (non-null labels)
    and their transaction edges. This significantly reduces memory usage since
    only ~0.03% of nodes have labels, making it much more manageable.
    
    Args:
        root: Root directory where the dataset is stored
        max_nodes: Maximum number of labeled nodes to include (None for all)
        seed: Random seed for reproducible subsampling
        
    Attributes:
        num_classes: Number of actual label classes (excluding UNLABELED)
        label_names: List of actual label names (no UNLABELED)
    """
    
    # All labels except UNLABELED for reduced dataset
    LABEL_NAMES = [
        "BET",           # 0
        "BRIDGE",        # 1  
        "EXCHANGE",      # 2
        "FAUCET",        # 3
        "INDIVIDUAL",    # 4
        "MARKETPLACE",   # 5
        "MINING",        # 6
        "MIXER",         # 7
        "PONZI",         # 8
        "RANSOMWARE",    # 9
        "OTHER"          # 10 - for any other labels
    ]
    
    def __init__(
        self,
        root: Optional[str] = None,
        max_nodes: Optional[int] = None,
        seed: int = 42,
    ):
        # Initialize parent but override subset behavior
        super().__init__(root=root, subset="labeled", max_nodes=max_nodes, seed=seed)
    
    @property
    def num_classes(self) -> int:
        """Number of label classes (excluding UNLABELED)."""
        return len(self.LABEL_NAMES)
    
    def _encode_label(self, label_str: Optional[str]) -> int:
        """Convert string label to integer (reduced mapping)."""
        if label_str is None or label_str == "null":
            raise ValueError("Reduced dataset should not have null labels")
        
        try:
            return self.LABEL_NAMES.index(label_str)
        except ValueError:
            # Unknown label -> OTHER
            return len(self.LABEL_NAMES) - 1
    
    def _load_nodes(self) -> tuple[pl.DataFrame, dict[int, int]]:
        """Load and process only labeled node features."""
        print(f"Loading labeled nodes from {self.nodes_file}")
        
        # Select relevant columns and filter for labeled nodes only
        nodes_lazy = pl.scan_parquet(self.nodes_file).select([
            'alias',
            'degree', 'degree_in', 'degree_out',
            'total_transactions_in', 'total_transactions_out', 
            'total_sent', 'total_received',
            'cluster_size',
            'label'
        ]).filter(
            pl.col('label').is_not_null()  # Only labeled nodes
        )
        
        # Apply subsampling if requested
        if self.max_nodes is not None:
            print(f"Sampling {self.max_nodes} labeled nodes...")
            nodes = nodes_lazy.collect().sample(n=self.max_nodes, seed=self.seed)
        else:
            nodes = nodes_lazy.collect()
        
        print(f"Loaded {len(nodes)} labeled nodes")
        
        # Create node ID mapping (alias -> consecutive IDs starting from 0)
        unique_aliases = nodes['alias'].unique().sort()
        alias_to_id = {alias: i for i, alias in enumerate(unique_aliases)}
        
        return nodes, alias_to_id
    
    def _create_labels(self, nodes: pl.DataFrame) -> torch.Tensor:
        """Create label tensor (using reduced label mapping)."""
        # Convert string labels to integers using reduced mapping
        labels = []
        for label_str in nodes['label'].to_list():
            labels.append(self._encode_label(label_str))
        
        return torch.tensor(labels, dtype=torch.long)
    
    def get(self, idx: int = 0) -> Data:
        """Get the reduced Bitcoin transaction graph data."""
        if idx != 0:
            raise IndexError(f"BTCReducedDataset only has one graph (idx=0), got {idx}")
        
        print("Loading reduced Bitcoin transaction graph (labeled nodes only)...")
        
        # Load labeled nodes and create mapping
        nodes, alias_to_id = self._load_nodes()
        
        # Load edges between labeled nodes
        edges = self._load_edges(alias_to_id)
        
        # Create PyTorch Geometric data
        print("Creating PyTorch Geometric data...")
        
        x = self._create_node_features(nodes)
        y = self._create_labels(nodes)  # Using reduced label mapping
        edge_index = self._create_edge_index(edges, alias_to_id)
        
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            num_nodes=len(nodes)
        )
        
        # Add metadata
        data.num_classes = self.num_classes
        data.label_names = self.label_names
        
        print(f"Created reduced graph with {data.num_nodes:,} labeled nodes, {data.edge_index.size(1):,} edges")
        print(f"Node features: {data.x.shape}")
        print(f"Classes: {data.num_classes} ({', '.join(data.label_names)})")
        
        # Print label distribution
        unique_labels, counts = torch.unique(data.y, return_counts=True)
        print("\nLabel distribution:")
        for label_id, count in zip(unique_labels.tolist(), counts.tolist()):
            label_name = data.label_names[label_id]
            print(f"  {label_name}: {count:,} ({count/len(data.y)*100:.1f}%)")
        
        return data
    
    def __repr__(self) -> str:
        return (f"BTCReducedDataset("
                f"root={self.root}, "
                f"max_nodes={self.max_nodes}, "
                f"labeled_only=True)")


# For backward compatibility
BtcDataset = BTCDataset
BtcReducedDataset = BTCReducedDataset