"""
Patched Graph Embedding Implementation

This module provides patched embedding methods that decompose graphs into
patches, embed them locally, and align them globally.
"""

import warnings
from typing import Any

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data

from ...graphs import TGraph
from ...patch import create_patches
from ..base import GraphEmbedding
from ..registry import get_embedding, register_embedding


@register_embedding("patched", aliases=["patch", "l2g_pipeline"])
class PatchedEmbedding(GraphEmbedding):
    """
    Patched embedding method that decomposes graphs and aligns local embeddings.

    This method creates patches of the input graph, embeds each patch independently,
    and then aligns them using configurable alignment methods.

    Args:
        embedding_dim: Dimensionality of the embedding
        aligner: Pre-configured aligner object (required)
        num_patches: Number of patches to create
        base_method: Base embedding method for patches (default: "vgae")
        clustering_method: Method for graph partitioning (default: "metis")
        min_overlap: Minimum overlap between patches
        target_overlap: Target overlap during patch expansion
        sparsify_method: Method for patch graph sparsification
        target_patch_degree: Target degree for patch graph
        verbose: Enable verbose output
        **embedding_kwargs: Additional arguments for base embedding method

    Examples:
        ```python
        from l2gx.align import get_aligner

        # Example 1: L2G aligner
        l2g_aligner = get_aligner("l2g")
        embedder = PatchedEmbedding(
            embedding_dim=128,
            aligner=l2g_aligner,
            num_patches=10,
            base_method="vgae",
            epochs=500
        )

        # Example 2: Geo aligner
        geo_aligner = get_aligner("geo")
        embedder = PatchedEmbedding(
            embedding_dim=128,
            aligner=geo_aligner,
            num_patches=8,
            base_method="svd",
        )

        # Example 3: L2G with custom settings
        l2g_aligner = get_aligner("l2g")
        l2g_aligner.randomized_method = "randomized"
        l2g_aligner.sketch_method = "gaussian"

        embedder = PatchedEmbedding(
            embedding_dim=64,
            aligner=l2g_aligner,
            num_patches=12,
            base_method="dgi"
        )

        # Example 4: Geo aligner with custom settings
        geo_aligner = get_aligner("geo", 
            method="euclidean",
            use_scale=False,
            num_epochs=10
        )

        embedder = PatchedEmbedding(
            embedding_dim=96,
            aligner=geo_aligner,
            num_patches=8,
            base_method="graphsage"
        )
        ```
    """

    def __init__(
        self,
        embedding_dim: int,
        aligner: Any,  # Required aligner object
        num_patches: int = 10,
        base_method: str = "vgae",
        clustering_method: str = "metis",
        min_overlap: int = 10,
        target_overlap: int = 20,
        sparsify_method: str = "resistance",
        target_patch_degree: int = 4,
        verbose: bool = False,
        **embedding_kwargs
    ):
        super().__init__(embedding_dim, **embedding_kwargs)

        self.num_patches = num_patches
        self.base_method = base_method
        self.clustering_method = clustering_method
        self.min_overlap = min_overlap
        self.target_overlap = target_overlap
        self.sparsify_method = sparsify_method
        self.target_patch_degree = target_patch_degree
        self.verbose = verbose
        self.embedding_kwargs = embedding_kwargs

        # Store aligner (required)
        if aligner is None:
            raise ValueError("An aligner object is required. Use get_aligner() to create one.")

        self.aligner = aligner
        self.alignment_method = getattr(aligner, 'name', 'custom')

        # Try to infer alignment method from class name if not available
        if hasattr(self.aligner, '__class__'):
            class_name = self.aligner.__class__.__name__.lower()
            if 'l2g' in class_name:
                self.alignment_method = "l2g"
            elif 'geo' in class_name:
                self.alignment_method = "geo"

        # Store patch information
        self._patches = None
        self._patch_graph = None
        self._tgraph = None
        self._global_embedding = None

    def fit(self, data: Data | nx.Graph | sp.spmatrix | np.ndarray, verbose: bool = False) -> 'PatchedEmbedding':
        """
        Fit the patched embedding model.

        Args:
            data: Input graph
            verbose: Unused, patched embedder uses its own verbose setting

        Returns:
            Self for method chaining
        """
        # Convert to TGraph if needed
        if isinstance(data, Data):
            self._tgraph = TGraph.from_tg(data)
        elif isinstance(data, nx.Graph):
            # Convert NetworkX to TGraph
            import torch
            edge_index = np.array(list(data.edges())).T
            if edge_index.size == 0:
                edge_index = np.empty((2, 0), dtype=int)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            pg_data = Data(edge_index=edge_index, num_nodes=data.number_of_nodes())
            self._tgraph = TGraph.from_tg(pg_data)
        elif isinstance(data, sp.spmatrix | np.ndarray):
            # Convert matrix to TGraph
            import torch
            if isinstance(data, np.ndarray):
                data = sp.csr_matrix(data)
            edge_index = torch.tensor(np.array(data.nonzero()), dtype=torch.long)
            pg_data = Data(edge_index=edge_index, num_nodes=data.shape[0])
            self._tgraph = TGraph.from_tg(pg_data)
        else:
            self._tgraph = data

        if self.verbose:
            print(f"Processing graph with {self._tgraph.num_nodes} nodes and {self._tgraph.num_edges} edges")

        # Create patches
        self._create_patches()

        # Embed patches
        self._embed_patches()

        # Align patches
        self._align_patches()

        return self

    def transform(self, data: Data | nx.Graph | sp.spmatrix | np.ndarray) -> np.ndarray:
        """
        Transform the graph into embeddings.

        Args:
            data: Input graph (unused for fitted model)

        Returns:
            Node embeddings as numpy array
        """
        if self._global_embedding is None:
            raise ValueError("Model must be fitted before transform")

        return self._global_embedding

    def _create_patches(self):
        """Create overlapping patches of the graph."""
        if self.verbose:
            print(f"\nCreating {self.num_patches} patches using {self.clustering_method}...")

        self._patch_graph = create_patches(
            self._tgraph,
            num_patches=self.num_patches,
            clustering_method=self.clustering_method,
            min_overlap=self.min_overlap,
            target_overlap=self.target_overlap,
            sparsify_method=self.sparsify_method,
            target_patch_degree=self.target_patch_degree,
            verbose=False  # Use our own verbose handling
        )

        self._patches = self._patch_graph.patches

        if self.verbose:
            patch_sizes = [len(patch.nodes) for patch in self._patches]
            print(f"Created {len(self._patches)} patches")
            print(f"  Sizes: min={min(patch_sizes)}, max={max(patch_sizes)}, avg={np.mean(patch_sizes):.1f}")

    def _embed_patches(self):
        """Embed each patch using the base embedding method."""
        if self.verbose:
            print(f"\nEmbedding {len(self._patches)} patches using {self.base_method}...")

        # Filter out parameters that conflict with patch-specific ones
        base_kwargs = {k: v for k, v in self.embedding_kwargs.items()
                       if k not in {'verbose', 'aligner', 'alignment_method',
                                    'num_patches', 'clustering_method', 'min_overlap',
                                    'target_overlap', 'sparsify_method', 'target_patch_degree'}}

        # Create base embedder
        embedder = get_embedding(
            self.base_method,
            embedding_dim=self.embedding_dim,
            **base_kwargs
        )

        # Embed each patch
        for i, patch in enumerate(self._patches):
            if self.verbose and i % max(1, len(self._patches) // 10) == 0:
                print(f"  Embedding patch {i+1}/{len(self._patches)}")

            # Extract subgraph for this patch
            patch_nodes = torch.tensor(patch.nodes, dtype=torch.long)
            patch_subgraph = self._tgraph.subgraph(patch_nodes, relabel=True)

            # Convert to PyTorch Geometric format
            pg_data = patch_subgraph.to_tg()

            # Compute embedding (use fit + transform separately to avoid verbose issues)
            patch.coordinates = embedder.fit(pg_data).transform(pg_data)

    def _align_patches(self):
        """Align patch embeddings to create global embedding."""
        if self.verbose:
            print(f"\nAligning patches using {self.alignment_method}...")

        if self.alignment_method == "procrustes" and len(self._patches) > 2:
            # Special handling for Procrustes with multiple patches
            self._global_embedding = self._align_procrustes_sequential()
        else:
            # Use configured aligner
            self.aligner.align_patches(self._patch_graph)
            self._global_embedding = self.aligner.get_aligned_embedding()

        if self.verbose:
            print(f"Global embedding shape: {self._global_embedding.shape}")

    def _align_procrustes_sequential(self) -> np.ndarray:
        """Sequential pairwise Procrustes alignment for multiple patches."""
        from scipy.spatial import procrustes

        # Initialize with first patch
        global_embedding = np.zeros((self._tgraph.num_nodes, self.embedding_dim))
        embedded_mask = np.zeros(self._tgraph.num_nodes, dtype=bool)

        # Place first patch
        first_patch = self._patches[0]
        global_embedding[first_patch.nodes] = first_patch.coordinates
        embedded_mask[first_patch.nodes] = True

        # Sequentially align remaining patches
        for patch in self._patches[1:]:
            # Find overlap with already embedded nodes
            overlap_global = np.intersect1d(patch.nodes, np.where(embedded_mask)[0])

            if len(overlap_global) >= 3:  # Need at least 3 points for Procrustes
                # Get overlapping embeddings
                overlap_mask_patch = np.isin(patch.nodes, overlap_global)
                patch_overlap = patch.coordinates[overlap_mask_patch]
                global_overlap = global_embedding[overlap_global]

                # Align using Procrustes
                _, aligned_patch, disparity = procrustes(global_overlap, patch_overlap)

                # Apply transformation to entire patch
                scale = np.linalg.norm(aligned_patch) / np.linalg.norm(patch_overlap)
                R = aligned_patch.T @ patch_overlap / (np.linalg.norm(patch_overlap) ** 2)
                aligned_coordinates = patch.coordinates @ R * scale

                # Update global embedding (average overlapping regions)
                for i, node_idx in enumerate(patch.nodes):
                    if embedded_mask[node_idx]:
                        # Average with existing embedding
                        global_embedding[node_idx] = (global_embedding[node_idx] + aligned_coordinates[i]) / 2
                    else:
                        global_embedding[node_idx] = aligned_coordinates[i]
                        embedded_mask[node_idx] = True
            else:
                # Not enough overlap, place without alignment (shouldn't happen with good patches)
                warnings.warn(f"Patch has insufficient overlap ({len(overlap_global)} nodes)")
                global_embedding[patch.nodes] = patch.coordinates
                embedded_mask[patch.nodes] = True

        return global_embedding

    def get_patches(self) -> list | None:
        """Get the computed patches after fitting."""
        return self._patches

    def get_patch_graph(self):
        """Get the patch graph structure after fitting."""
        return self._patch_graph

    def get_patch_info(self) -> dict[str, Any]:
        """Get information about the patch structure."""
        if self._patches is None:
            return {"fitted": False}

        patch_sizes = [len(patch.nodes) for patch in self._patches]

        # Calculate overlap statistics
        overlaps = []
        for i, patch1 in enumerate(self._patches):
            for patch2 in self._patches[i+1:]:
                overlap = len(np.intersect1d(patch1.nodes, patch2.nodes))
                if overlap > 0:
                    overlaps.append(overlap)

        return {
            "fitted": True,
            "num_patches": len(self._patches),
            "patch_sizes": {
                "min": min(patch_sizes),
                "max": max(patch_sizes),
                "mean": np.mean(patch_sizes),
                "std": np.std(patch_sizes)
            },
            "overlaps": {
                "count": len(overlaps),
                "min": min(overlaps) if overlaps else 0,
                "max": max(overlaps) if overlaps else 0,
                "mean": np.mean(overlaps) if overlaps else 0
            },
            "clustering_method": self.clustering_method,
            "alignment_method": self.alignment_method
        }
