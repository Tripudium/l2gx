"""
Elliptic Bitcoin Dataset.

The Elliptic dataset maps Bitcoin transactions to real entities belonging to 
licit categories (exchanges, wallet providers, miners, licit services, etc.) 
versus illicit ones (scams, malware, terrorist organizations, ransomware, 
Ponzi schemes, etc.).

Reference: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
Paper: Weber et al. "Anti-Money Laundering in Bitcoin: Experimenting with Graph
Convolutional Networks for Financial Forensics" (2019)
"""

import zipfile
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Optional, Callable
from .registry import register_dataset
from .base import BaseDataset


@register_dataset("Elliptic")
class EllipticDataset(BaseDataset):
    """
    Elliptic Bitcoin transaction dataset for illicit transaction detection.
    
    The dataset contains 203,769 nodes representing Bitcoin transactions and 
    234,355 directed edges representing payment flows. Each node has 166 features
    and is labeled as licit (1), illicit (2), or unknown (0).
    """
    
    def __init__(
        self,
        root: str | None = None,
        source_file: str | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        """
        Initialize the Elliptic dataset.
        
        Args:
            root: Root directory for dataset storage
            source_file: Path to the downloaded elliptic_bitcoin_dataset.zip file
            transform: Optional data transformation function
            pre_transform: Optional preprocessing transformation function
            
        Note:
            The dataset must be manually downloaded from Kaggle and the path
            provided via the source_file parameter.
        """
        # Logger is set up in BaseDataset
        
        if root is None:
            root = str(Path(__file__).parent.parent.parent / "data" / "elliptic")
        
        self.source_file = source_file
        self._raw_paths = [source_file] if source_file else []
        
        super().__init__(root, transform, pre_transform)
        
        # Load processed data
        self.edge_df, self.node_df = self._load_polars()
        self.data, self.slices = self._to_torch_geometric()
        self.raphtory_graph = self._to_raphtory()

    @property
    def raw_file_names(self) -> list[str]:
        """Required raw files for the Elliptic dataset."""
        return ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        """Processed file names."""
        return ["edge_data.parquet", "node_data.parquet"]

    def download(self):
        """
        Extract the Elliptic dataset from the provided zip file.
        """
        if self.source_file is None:
            raise ValueError(
                "source_file must be provided. Download the elliptic_bitcoin_dataset.zip "
                "from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
            )
        
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        source_file = Path(self.source_file)
        
        if not source_file.exists():
            raise FileNotFoundError(f"Source file {self.source_file} not found")
        
        # Check if files already extracted
        required_files = [raw_dir / fname for fname in self.raw_file_names]
        if all(f.exists() for f in required_files):
            self.logger.info("Dataset files already extracted")
            return
        
        self.logger.info(f"Extracting dataset from {source_file}")
        with zipfile.ZipFile(source_file, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        self.logger.info("Dataset extraction complete")

    def process(self):
        """
        Process the raw CSV files into Polars DataFrames.
        """
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already processed
        if all((processed_dir / fname).exists() for fname in self.processed_file_names):
            self.logger.info("Dataset already processed")
            return
        
        raw_dir = Path(self.raw_dir)
        
        self.logger.info("Processing Elliptic dataset...")
        
        # Load node features and classes
        features_df = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
        classes_df = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
        edges_df = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
        
        # Process node data
        # First column is txId, second is time step, rest are features
        node_features = features_df.iloc[:, 2:].values  # Features (columns 2-166)
        node_ids = features_df.iloc[:, 0].values        # Transaction IDs
        time_steps = features_df.iloc[:, 1].values      # Time steps
        
        # Create node mapping from txId to integer index
        unique_nodes = sorted(set(node_ids) | set(edges_df['txId1']) | set(edges_df['txId2']))
        node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
        
        # Merge with classes (labels)
        classes_dict = dict(zip(classes_df['txId'], classes_df['class']))
        
        # Create node dataframe
        node_data = []
        for i, (tx_id, time_step) in enumerate(zip(node_ids, time_steps)):
            node_idx = node_to_idx[tx_id]
            label = classes_dict.get(tx_id, '0')  # 0 = unknown, 1 = licit, 2 = illicit
            
            # Convert label to numeric
            if label == 'unknown':
                label_num = 0
            elif label == '1':
                label_num = 1
            elif label == '2':
                label_num = 2
            else:
                label_num = 0
            
            row = {
                'id': node_idx,
                'timestamp': time_step,
                'tx_id': tx_id,
                'y': label_num,
            }
            
            # Add features
            if i < len(node_features):
                for j, feat_val in enumerate(node_features[i]):
                    row[f'feature_{j}'] = feat_val
            
            node_data.append(row)
        
        node_df = pl.DataFrame(node_data)
        
        # Process edge data
        edge_data = []
        for _, row in edges_df.iterrows():
            src_idx = node_to_idx.get(row['txId1'])
            dst_idx = node_to_idx.get(row['txId2'])
            
            if src_idx is not None and dst_idx is not None:
                edge_data.append({
                    'src': src_idx,
                    'dst': dst_idx,
                    'timestamp': 0,  # Static graph
                })
        
        edge_df = pl.DataFrame(edge_data)
        
        # Save processed data
        edge_df.write_parquet(processed_dir / "edge_data.parquet")
        node_df.write_parquet(processed_dir / "node_data.parquet")
        
        self.logger.info(f"Processing complete. Saved {len(node_df)} nodes and {len(edge_df)} edges")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes=203769, edges=234355)"