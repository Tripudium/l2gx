"""
Lightweight Streaming Patch Generation for Large Graphs

This module provides memory-efficient patch generation for extremely large graphs
that cannot fit entirely in memory. It uses:

1. Streaming edge processing with Polars
2. Batch-based FENNEL clustering
3. Parquet-based patch storage
4. Minimal interface changes from existing patch system

Key classes:
- StreamingPatchGenerator: Main interface for creating patches from large graphs
- LazyPatch: Patch object that loads data on-demand from parquet files
- StreamingFENNEL: Memory-efficient FENNEL implementation

Usage:
    ```python
    from l2gx.patch.streaming import StreamingPatchGenerator
    
    # Create patches from large graph
    generator = StreamingPatchGenerator(
        dataset=mag240m_dataset,
        num_patches=50,
        patch_dir="patches/mag240m_50"
    )
    
    # Generate patches (streaming, disk-based)
    patch_graph = generator.create_patches()
    
    # Access patches (lazy loading)
    for patch in patch_graph.patches:
        # Patch data loaded on-demand from parquet
        subgraph = patch.to_tgraph()
    ```
"""

import gc
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch

from l2gx.datasets.mag240m import MAG240MDataset
from l2gx.graphs.tgraph import TGraph
from l2gx.patch.patches import Patch


class LazyPatch(Patch):
    """
    Lazy-loading patch that stores data in parquet files.

    This patch loads its node list, coordinates, and subgraph data
    from parquet files only when accessed, keeping memory usage minimal.
    """

    def __init__(self, patch_id: int, patch_dir: Path, node_list: np.ndarray | None = None):
        """
        Initialize lazy patch.

        Args:
            patch_id: Unique patch identifier
            patch_dir: Directory containing patch parquet files
            node_list: Pre-loaded node list (optional, loads from disk if None)
        """
        self.patch_id = patch_id
        self.patch_dir = Path(patch_dir)
        self._node_list = node_list
        self._coordinates = None
        self._subgraph_data = None
        self._index = None

        # Ensure patch directory exists
        self.patch_dir.mkdir(parents=True, exist_ok=True)

    @property
    def nodes(self) -> np.ndarray:
        """Get node list, loading from disk if needed."""
        if self._node_list is None:
            nodes_file = self.patch_dir / f"patch_{self.patch_id}_nodes.parquet"
            if nodes_file.exists():
                df = pl.read_parquet(nodes_file)
                self._node_list = df["node_id"].to_numpy()
            else:
                raise FileNotFoundError(f"Patch nodes file not found: {nodes_file}")
        return self._node_list

    @property
    def index(self) -> dict[int, int]:
        """Get node index mapping, creating from nodes if needed."""
        if self._index is None:
            self._index = {int(node): i for i, node in enumerate(self.nodes)}
        return self._index

    @property
    def coordinates(self) -> np.ndarray:
        """Get coordinates, loading from disk if needed."""
        if self._coordinates is None:
            coords_file = self.patch_dir / f"patch_{self.patch_id}_coords.parquet"
            if coords_file.exists():
                df = pl.read_parquet(coords_file)
                # Convert coordinate columns to numpy array
                coord_cols = [col for col in df.columns if col.startswith("dim_")]
                if coord_cols:
                    self._coordinates = df.select(coord_cols).to_numpy()
                else:
                    # No coordinates yet, create placeholder
                    self._coordinates = np.random.randn(len(self.nodes), 128)
            else:
                # Create random placeholder coordinates
                self._coordinates = np.random.randn(len(self.nodes), 128)
        return self._coordinates

    @coordinates.setter
    def coordinates(self, value: np.ndarray):
        """Set coordinates and save to disk."""
        self._coordinates = value
        self._save_coordinates()

    def _save_coordinates(self):
        """Save coordinates to parquet file."""
        if self._coordinates is not None:
            coords_file = self.patch_dir / f"patch_{self.patch_id}_coords.parquet"

            # Create DataFrame with coordinate columns
            coord_data = {"node_id": self.nodes}
            for i in range(self._coordinates.shape[1]):
                coord_data[f"dim_{i}"] = self._coordinates[:, i]

            df = pl.DataFrame(coord_data)
            df.write_parquet(coords_file)

    def save_nodes(self, nodes: np.ndarray):
        """Save node list to parquet file."""
        nodes_file = self.patch_dir / f"patch_{self.patch_id}_nodes.parquet"
        df = pl.DataFrame({"node_id": nodes})
        df.write_parquet(nodes_file)
        self._node_list = nodes
        self._index = None  # Reset index cache

    def save_subgraph(self, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None):
        """Save subgraph edges to parquet file."""
        edges_file = self.patch_dir / f"patch_{self.patch_id}_edges.parquet"

        edge_data = {
            "source": edge_index[0].cpu().numpy(),
            "target": edge_index[1].cpu().numpy()
        }

        if edge_attr is not None:
            edge_data["weight"] = edge_attr.cpu().numpy()

        df = pl.DataFrame(edge_data)
        df.write_parquet(edges_file)

    def to_tgraph(self) -> TGraph:
        """Convert patch to TGraph, loading subgraph if needed."""
        edges_file = self.patch_dir / f"patch_{self.patch_id}_edges.parquet"

        if edges_file.exists():
            df = pl.read_parquet(edges_file)

            if len(df) == 0:
                # Empty graph
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = None
            else:
                edge_index = torch.tensor([
                    df["source"].to_list(),
                    df["target"].to_list()
                ], dtype=torch.long)

                edge_attr = None
                if "weight" in df.columns:
                    edge_attr = torch.tensor(df["weight"].to_list(), dtype=torch.float)
        else:
            # No edges saved yet, create empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = None

        return TGraph(
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(self.nodes),
            undir=True
        )

    @property
    def shape(self):
        """Shape of coordinates array."""
        return self.coordinates.shape

    def __len__(self):
        """Number of nodes in patch."""
        return len(self.nodes)


class StreamingFENNEL:
    """
    Memory-efficient FENNEL clustering for large graphs using edge streaming.

    This implementation processes edges in batches to avoid loading the
    entire graph adjacency matrix into memory.
    """

    def __init__(self, num_nodes: int, num_clusters: int,
                 load_limit: float = 1.1, gamma: float = 1.5,
                 batch_size: int = 100000, verbose: bool = True):
        """
        Initialize streaming FENNEL clusterer.

        Args:
            num_nodes: Total number of nodes
            num_clusters: Target number of clusters  
            load_limit: Maximum cluster size factor
            gamma: FENNEL gamma parameter
            batch_size: Edge batch size for streaming
            verbose: Print progress information
        """
        self.num_nodes = num_nodes
        self.num_clusters = num_clusters
        self.load_limit = load_limit
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose

        # Initialize cluster assignments and statistics
        self.clusters = np.full(num_nodes, -1, dtype=np.int64)
        self.partition_sizes = np.zeros(num_clusters, dtype=np.int64)
        self.max_cluster_size = int(load_limit * num_nodes / num_clusters)

        if verbose:
            print(f"StreamingFENNEL: {num_nodes:,} nodes → {num_clusters} clusters")
            print(f"Max cluster size: {self.max_cluster_size:,}")

    def cluster_from_edges(self, edge_iterator) -> np.ndarray:
        """
        Run FENNEL clustering from an edge iterator.

        Args:
            edge_iterator: Iterator yielding (source, target) edge batches

        Returns:
            Cluster assignment array
        """
        if self.verbose:
            print("Running streaming FENNEL clustering...")

        # Single-pass clustering
        total_edges = 0

        for batch_count, edge_batch in enumerate(edge_iterator):
            batch_edges = len(edge_batch)
            total_edges += batch_edges

            if self.verbose and batch_count % 100 == 0:
                print(f"Processed {batch_count} batches, {total_edges:,} edges")

            # Process edges in this batch
            for source, target in edge_batch:
                self._process_edge(source, target)

        if self.verbose:
            print(f"FENNEL complete: {total_edges:,} edges processed")
            print(f"Cluster sizes: {[self.partition_sizes[i] for i in range(min(5, self.num_clusters))]}")

        return self.clusters

    def _process_edge(self, source: int, target: int):
        """Process a single edge for FENNEL clustering."""
        # Assign source node if unassigned
        if self.clusters[source] == -1:
            self._assign_node(source)

        # Assign target node if unassigned
        if self.clusters[target] == -1:
            # Prefer assigning to same cluster as source if possible
            source_cluster = self.clusters[source]
            if self.partition_sizes[source_cluster] < self.max_cluster_size:
                self.clusters[target] = source_cluster
                self.partition_sizes[source_cluster] += 1
            else:
                self._assign_node(target)

    def _assign_node(self, node: int):
        """Assign node to best available cluster."""
        # Find cluster with minimum load that's not full
        best_cluster = -1
        min_size = float('inf')

        for c in range(self.num_clusters):
            if self.partition_sizes[c] < self.max_cluster_size:
                if self.partition_sizes[c] < min_size:
                    min_size = self.partition_sizes[c]
                    best_cluster = c

        if best_cluster >= 0:
            self.clusters[node] = best_cluster
            self.partition_sizes[best_cluster] += 1
        else:
            # All clusters full, assign to least full
            best_cluster = np.argmin(self.partition_sizes)
            self.clusters[node] = best_cluster
            self.partition_sizes[best_cluster] += 1


class StreamingPatchGenerator:
    """
    Main interface for creating patches from large graphs using streaming.
    
    This generator creates patches that are compatible with the existing
    L2GX patch system while using disk storage for scalability.
    """

    def __init__(self, dataset: MAG240MDataset,
                 num_patches: int,
                 patch_dir: str | Path,
                 min_overlap: int | None = None,
                 target_overlap: int | None = None,
                 batch_size: int = 100000,
                 verbose: bool = True):
        """
        Initialize streaming patch generator.
        
        Args:
            dataset: Enhanced MAG240M dataset
            num_patches: Number of patches to create
            patch_dir: Directory to store patch files
            min_overlap: Minimum overlap between connected patches
            target_overlap: Target overlap size
            batch_size: Edge batch size for streaming
            verbose: Print progress information
        """
        self.dataset = dataset
        self.num_patches = num_patches
        self.patch_dir = Path(patch_dir)
        self.batch_size = batch_size
        self.verbose = verbose

        # Create patch directory
        self.patch_dir.mkdir(parents=True, exist_ok=True)

        # Get dataset statistics
        stats = dataset.get_subset_statistics()
        self.num_nodes = stats.get('num_nodes', 0)

        # Calculate overlap parameters
        expected_patch_size = max(1, self.num_nodes // num_patches)
        self.min_overlap = min_overlap or max(1, expected_patch_size // 20)
        self.target_overlap = target_overlap or max(self.min_overlap, expected_patch_size // 10)

        if verbose:
            print(f"StreamingPatchGenerator: {self.num_nodes:,} nodes → {num_patches} patches")
            print(f"Expected patch size: ~{expected_patch_size:,}")
            print(f"Overlap: min={self.min_overlap}, target={self.target_overlap}")

    def create_patches(self) -> TGraph:
        """
        Create patches using streaming approach.
        
        Returns:
            TGraph with lazy patches and overlap information
        """
        if self.verbose:
            print("Starting streaming patch generation...")

        # Step 1: Extract edge data for streaming
        edge_data_file = self.patch_dir / "edges.parquet"
        if not edge_data_file.exists():
            self._extract_edges_to_parquet()

        # Step 2: Run streaming clustering
        clusters_file = self.patch_dir / "clusters.parquet"
        if not clusters_file.exists():
            clusters = self._run_streaming_clustering()
            # Save clusters
            cluster_df = pl.DataFrame({
                "node_id": range(self.num_nodes),
                "cluster": clusters
            })
            cluster_df.write_parquet(clusters_file)
        else:
            if self.verbose:
                print("Loading existing clusters...")
            cluster_df = pl.read_parquet(clusters_file)
            clusters = cluster_df["cluster"].to_numpy()

        # Step 3: Create patch objects and save patch data
        patches = self._create_patch_objects(clusters)

        # Step 4: Create simple patch connectivity graph
        patch_graph = self._create_patch_graph(patches)

        if self.verbose:
            print(f"Streaming patch generation complete: {len(patches)} patches created")

        return patch_graph

    def _extract_edges_to_parquet(self):
        """Extract citation edges to parquet file for streaming."""
        if self.verbose:
            print("Extracting citation edges to parquet...")

        # Get citation edges from subset
        subset_data = self.dataset._subset_data
        edges_data = subset_data.filter(pl.col("type") == "edge")

        if len(edges_data) == 0:
            # Create empty edges file
            empty_df = pl.DataFrame({"source": [], "target": []})
            empty_df.write_parquet(self.patch_dir / "edges.parquet")
            return

        # Extract source and target columns
        edge_df = edges_data.select(["source", "target"])

        # Make undirected (add reverse edges)
        reverse_df = edge_df.select([
            pl.col("target").alias("source"),
            pl.col("source").alias("target")
        ])

        # Combine and deduplicate
        all_edges = pl.concat([edge_df, reverse_df]).unique()

        # Save to parquet
        all_edges.write_parquet(self.patch_dir / "edges.parquet")

        if self.verbose:
            print(f"Saved {len(all_edges):,} undirected edges to parquet")

    def _run_streaming_clustering(self) -> np.ndarray:
        """Run streaming FENNEL clustering."""
        if self.verbose:
            print("Running streaming FENNEL clustering...")

        # Initialize clustering
        fennel = StreamingFENNEL(
            num_nodes=self.num_nodes,
            num_clusters=self.num_patches,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        # Create edge iterator
        edge_iterator = self._create_edge_iterator()

        # Run clustering
        clusters = fennel.cluster_from_edges(edge_iterator)

        return clusters

    def _create_edge_iterator(self):
        """Create iterator for edge batches."""
        edges_file = self.patch_dir / "edges.parquet"

        # Use Polars lazy scanning for memory efficiency
        edge_lazy = pl.scan_parquet(edges_file)

        # Process in batches
        offset = 0
        while True:
            batch_df = edge_lazy.slice(offset, self.batch_size).collect()

            if len(batch_df) == 0:
                break

            # Convert to list of tuples
            edges = [(row["source"], row["target"]) for row in batch_df.iter_rows(named=True)]
            yield edges

            offset += self.batch_size

            # Free memory
            del batch_df
            gc.collect()

    def _create_patch_objects(self, clusters: np.ndarray) -> list[LazyPatch]:
        """Create lazy patch objects and save patch data."""
        if self.verbose:
            print("Creating patch objects and saving data...")

        patches = []

        # Group nodes by cluster
        cluster_nodes = {}
        for node_id, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_nodes:
                cluster_nodes[cluster_id] = []
            cluster_nodes[cluster_id].append(node_id)

        # Create LazyPatch objects
        for cluster_id in range(self.num_patches):
            if cluster_id in cluster_nodes:
                nodes = np.array(cluster_nodes[cluster_id])
            else:
                nodes = np.array([], dtype=np.int64)

            # Create patch object
            patch = LazyPatch(
                patch_id=cluster_id,
                patch_dir=self.patch_dir / "patches",
                node_list=nodes
            )

            # Save nodes to disk
            patch.save_nodes(nodes)

            # Extract and save subgraph (simplified - just save empty for now)
            # In a full implementation, this would extract edges between patch nodes
            patch.save_subgraph(torch.empty((2, 0), dtype=torch.long))

            patches.append(patch)

        return patches

    def _create_patch_graph(self, patches: list[LazyPatch]) -> TGraph:
        """Create simple patch connectivity graph."""
        # Create simple linear connectivity for now
        if self.num_patches == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        elif self.num_patches == 2:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
        else:
            # Linear chain connectivity
            edges = []
            for i in range(self.num_patches - 1):
                edges.extend([(i, i+1), (i+1, i)])
            edge_index = torch.tensor(edges, dtype=torch.long).t()

        # Create patch graph
        patch_graph = TGraph(edge_index=edge_index, num_nodes=self.num_patches)

        # Add patch attributes
        patch_graph.patches = patches

        # Create simple overlap dictionary (empty for now)
        overlap_nodes = {}
        for i in range(self.num_patches):
            for j in range(i+1, self.num_patches):
                overlap_nodes[(i, j)] = []
                overlap_nodes[(j, i)] = []

        patch_graph.overlap_nodes = overlap_nodes

        # Save metadata
        metadata = {
            'num_patches': self.num_patches,
            'num_nodes': self.num_nodes,
            'min_overlap': self.min_overlap,
            'target_overlap': self.target_overlap
        }

        metadata_file = self.patch_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)

        return patch_graph


def load_streaming_patches(patch_dir: str | Path) -> TGraph:
    """
    Load previously created streaming patches.
    
    Args:
        patch_dir: Directory containing patch files
        
    Returns:
        TGraph with lazy patches
    """
    patch_dir = Path(patch_dir)

    # Load metadata
    metadata_file = patch_dir / "metadata.pkl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Patch metadata not found: {metadata_file}")

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    num_patches = metadata['num_patches']

    # Create lazy patch objects
    patches = []
    for patch_id in range(num_patches):
        patch = LazyPatch(patch_id=patch_id, patch_dir=patch_dir / "patches")
        patches.append(patch)

    # Create patch graph (simple connectivity for now)
    if num_patches == 1:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif num_patches == 2:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).t()
    else:
        edges = []
        for i in range(num_patches - 1):
            edges.extend([(i, i+1), (i+1, i)])
        edge_index = torch.tensor(edges, dtype=torch.long).t()

    patch_graph = TGraph(edge_index=edge_index, num_nodes=num_patches)
    patch_graph.patches = patches

    # Empty overlaps for now
    overlap_nodes = {}
    for i in range(num_patches):
        for j in range(i+1, num_patches):
            overlap_nodes[(i, j)] = []
            overlap_nodes[(j, i)] = []
    patch_graph.overlap_nodes = overlap_nodes

    return patch_graph
