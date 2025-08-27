"""
Hierarchical Graph Embedding Implementation

This module provides hierarchical embedding methods that recursively partition
graphs and align embeddings in a tree structure.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from ...align import get_aligner
from ...graphs import TGraph
from ...patch import create_patches
from ..base import GraphEmbedding
from ..registry import get_embedding, register_embedding


@dataclass
class HierarchicalNode:
    """Represents a node in the hierarchical tree."""
    graph: TGraph
    level: int
    node_indices: np.ndarray  # Original node indices in full graph
    embedding: np.ndarray | None = None
    is_leaf: bool = False
    children: list['HierarchicalNode'] = None
    parent: Optional['HierarchicalNode'] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


@register_embedding("hierarchical", aliases=["hier", "tree"])
class HierarchicalEmbedding(GraphEmbedding):
    """
    Hierarchical embedding method that recursively partitions and embeds graphs.

    This method creates a tree structure of graph partitions, embeds leaf nodes,
    and aligns embeddings bottom-up through the tree.

    Args:
        embedding_dim: Dimensionality of the embedding
        aligner: Pre-configured aligner object (required)
        max_patch_size: Maximum size for leaf patches
        min_patch_size: Minimum size for leaf patches (default: max_patch_size // 4)
        base_method: Base embedding method for leaf patches (default: "vgae")
        clustering_method: Method for hierarchical partitioning (default: "metis")
        branching_factor: Number of children per node (default: 2 for binary tree)
        max_levels: Maximum tree depth to prevent infinite recursion
        **embedding_kwargs: Additional arguments for base embedding method

    Examples:
        ```python
        from l2gx.align import get_aligner

        # Example 1: L2G aligner
        l2g_aligner = get_aligner("l2g")
        embedder = HierarchicalEmbedding(
            embedding_dim=128,
            aligner=l2g_aligner,
            max_patch_size=500,
            branching_factor=2,
        )

        # Example 2: Geo aligner with custom settings
        geo_aligner = get_aligner("geo", 
            method="euclidean", 
            use_scale=False,
            num_epochs=10,
            learning_rate=0.1
        )

        embedder = HierarchicalEmbedding(
            embedding_dim=128,
            aligner=geo_aligner,
            max_patch_size=500,
            branching_factor=3,
            base_method="svd"
        )

        # Example 3: L2G with randomization
        l2g_aligner = get_aligner("l2g")
        l2g_aligner.randomized_method = "randomized"
        l2g_aligner.sketch_method = "gaussian"

        embedder = HierarchicalEmbedding(
            embedding_dim=64,
            aligner=l2g_aligner,
            branching_factor=4,
        )
        ```
    """

    def __init__(
        self,
        embedding_dim: int,
        aligner: Any,  # Required aligner object
        max_patch_size: int = 500,
        min_patch_size: int | None = None,
        base_method: str = "vgae",
        clustering_method: str = "metis",
        branching_factor: int = 2,
        max_levels: int = 10,
        min_overlap: int = 64,
        target_overlap: int = 128,
        verbose: bool = False,
        **embedding_kwargs
    ):
        super().__init__(embedding_dim, **embedding_kwargs)

        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size or max_patch_size // 4
        self.base_method = base_method
        self.clustering_method = clustering_method
        self.branching_factor = branching_factor
        self.max_levels = max_levels
        self.min_overlap = min_overlap
        self.target_overlap = target_overlap
        self.verbose = verbose
        self.embedding_kwargs = embedding_kwargs

        # Store aligner and extract its configuration for creating fresh instances
        if aligner is None:
            raise ValueError("An aligner object is required. Use get_aligner() to create one.")
        
        self.aligner = aligner
        self.aligner_config = self._extract_aligner_config(aligner)
        self.alignment_method = getattr(aligner, 'name', self.aligner_config.get('type', 'procrustes'))
        
        # Use Procrustes by default for binary trees, otherwise use the provided aligner
        self.use_procrustes_for_binary = (branching_factor == 2)
        
        if self.use_procrustes_for_binary:
            if self.verbose:
                print(f"Using Procrustes alignment by default for binary tree (branching_factor={branching_factor})")
        else:
            if self.verbose:
                print(f"Using {self.alignment_method} alignment for {branching_factor}-ary tree")

        # Store tree structure
        self.root = None
        self.leaves = []

    def _extract_aligner_config(self, aligner: Any) -> dict[str, Any]:
        """Extract configuration parameters from an aligner instance for creating fresh instances."""
        config = {}
        
        # Determine aligner type from class name or registered name
        aligner_class_name = aligner.__class__.__name__.lower()
        if 'l2g' in aligner_class_name or hasattr(aligner, 'randomized_method'):
            config['type'] = 'l2g'
            config['randomized_method'] = getattr(aligner, 'randomized_method', 'standard')
            config['sketch_method'] = getattr(aligner, 'sketch_method', 'gaussian')
            config['verbose'] = getattr(aligner, 'verbose', False)
        elif 'geo' in aligner_class_name or hasattr(aligner, 'method'):
            config['type'] = 'geo'
            config['verbose'] = getattr(aligner, 'verbose', False)
            config['use_scale'] = getattr(aligner, 'use_scale', True)
            config['method'] = getattr(aligner, 'method', 'standard')
            config['orthogonal_reg_weight'] = getattr(aligner, 'orthogonal_reg_weight', 100.0)
            config['batch_size'] = getattr(aligner, 'batch_size', 512)
            config['use_bfs_training'] = getattr(aligner, 'use_bfs_training', True)
            config['device'] = getattr(aligner, 'device', 'cpu')
            config['patience'] = getattr(aligner, 'patience', 20)
            config['tolerance'] = getattr(aligner, 'tolerance', 1e-8)
            config['use_randomized_init'] = getattr(aligner, 'use_randomized_init', False)
            config['randomized_method'] = getattr(aligner, 'randomized_method', 'randomized')
            config['num_epochs'] = getattr(aligner, 'num_epochs', 1000)
            config['learning_rate'] = getattr(aligner, 'learning_rate', 0.01)
        else:
            # Default to Procrustes
            config['type'] = 'procrustes'
            config['verbose'] = getattr(aligner, 'verbose', False)
        
        return config

    def _create_fresh_aligner(self) -> Any:
        """Create a fresh aligner instance using the extracted configuration."""
        from ...align import get_aligner
        
        config = self.aligner_config.copy()
        aligner_type = config.pop('type')
        
        if aligner_type == 'l2g':
            return get_aligner('l2g', **config)
        elif aligner_type == 'geo':
            return get_aligner('geo', **config)
        else:
            # Fallback to a simple aligner that just does Procrustes
            return get_aligner('geo', method='orthogonal', use_scale=True, verbose=config.get('verbose', False))

    def fit(self, data: Data | TGraph, verbose: bool = False) -> 'HierarchicalEmbedding':
        """
        Fit the hierarchical embedding model.

        Args:
            data: Input graph (PyTorch Geometric Data or TGraph)
            verbose: Unused, hierarchical embedder uses its own verbose setting

        Returns:
            Self for method chaining
        """
        # Convert to TGraph if needed
        graph = TGraph.from_tg(data) if isinstance(data, Data) else data

        # Build hierarchical tree
        self.root = self._build_tree(graph, level=0)

        # Embed leaf patches
        self._embed_leaves()

        # Align embeddings bottom-up
        self._align_bottom_up(self.root)

        return self

    def transform(self, data: Data | TGraph) -> np.ndarray:
        """
        Transform the graph into embeddings.

        For hierarchical embedding, this returns the root level embedding
        which contains all nodes.

        Args:
            data: Input graph (unused for fitted model)

        Returns:
            Node embeddings as numpy array
        """
        if self.root is None or self.root.embedding is None:
            raise ValueError("Model must be fitted before transform")

        return self.root.embedding

    def _build_tree(self, graph: TGraph, level: int,
                    node_indices: np.ndarray | None = None) -> HierarchicalNode:
        """Recursively build the hierarchical tree structure."""

        if node_indices is None:
            node_indices = np.arange(graph.num_nodes)

        node = HierarchicalNode(
            graph=graph,
            level=level,
            node_indices=node_indices
        )

        # Check if this should be a leaf node
        if graph.num_nodes <= self.max_patch_size or level >= self.max_levels:
            node.is_leaf = True
            self.leaves.append(node)
            if self.verbose:
                print(f"  {'  ' * level}Leaf at level {level}: {graph.num_nodes} nodes")
        else:
            # Partition into child patches
            if self.verbose:
                print(f"  {'  ' * level}Partitioning level {level}: {graph.num_nodes} nodes")

            # Create overlapping patches
            patch_graph = create_patches(
                graph,
                num_patches=self.branching_factor,
                clustering_method=self.clustering_method,
                min_overlap=self.min_overlap,
                target_overlap=self.target_overlap,
                verbose=False
            )

            # Create child nodes
            for i, patch in enumerate(patch_graph.patches):
                child_indices = node_indices[patch.nodes]
                child_graph = graph.subgraph(torch.tensor(patch.nodes), relabel=True)

                child = self._build_tree(
                    child_graph,
                    level + 1,
                    child_indices
                )
                child.parent = node
                node.children.append(child)

        return node

    def _embed_leaves(self):
        """Embed all leaf patches using the base embedding method."""
        if self.verbose:
            print(f"\nEmbedding {len(self.leaves)} leaf patches...")

        # Filter out parameters that conflict with hierarchical-specific ones
        base_kwargs = {k: v for k, v in self.embedding_kwargs.items()
                       if k not in {'verbose', 'aligner', 'alignment_method',
                                    'branching_factor', 'max_levels', 'max_patch_size',
                                    'min_patch_size', 'clustering_method', 'min_overlap',
                                    'target_overlap'}}

        # Create base embedder
        embedder = get_embedding(
            self.base_method,
            embedding_dim=self.embedding_dim,
            **base_kwargs
        )

        # Embed each leaf
        for i, leaf in enumerate(self.leaves):
            if self.verbose:
                print(f"  Embedding leaf {i+1}/{len(self.leaves)} ({leaf.graph.num_nodes} nodes)")

            # Convert to PyTorch Geometric format
            pg_data = leaf.graph.to_tg()

            # Compute embedding (use fit + transform separately to avoid verbose parameter issues)
            leaf.embedding = embedder.fit(pg_data).transform(pg_data)

    def _align_bottom_up(self, node: HierarchicalNode):
        """Recursively align embeddings from leaves to root."""

        if node.is_leaf:
            # Leaf already has embedding
            return

        # First, ensure all children are aligned
        for child in node.children:
            self._align_bottom_up(child)

        if self.verbose:
            print(f"  Aligning level {node.level} ({node.graph.num_nodes} nodes)")

        # Choose alignment method based on configuration
        if self.use_procrustes_for_binary and len(node.children) == 2:
            # Use Procrustes for binary trees
            node.embedding = self._align_procrustes(node)
        elif len(node.children) == 2:
            # Always use Procrustes for 2 children (most stable)
            node.embedding = self._align_procrustes(node)
        else:
            # Use the configured aligner for multi-way splits
            try:
                node.embedding = self._align_with_aligner(node)
            except Exception as e:
                if self.verbose:
                    print(f"    Alignment failed ({e}), falling back to Procrustes")
                node.embedding = self._align_procrustes(node)

    def _align_procrustes(self, node: HierarchicalNode) -> np.ndarray:
        """Align child embeddings using Procrustes alignment."""

        # Initialize combined embedding
        embedding = np.zeros((node.graph.num_nodes, self.embedding_dim))
        counts = np.zeros(node.graph.num_nodes)

        # Use first child as reference
        ref_child = node.children[0]
        ref_embedding = ref_child.embedding

        # Map reference child nodes to parent
        for local_idx, global_idx in enumerate(ref_child.node_indices):
            parent_idx = np.where(node.node_indices == global_idx)[0][0]
            embedding[parent_idx] = ref_embedding[local_idx]
            counts[parent_idx] += 1

        # Align other children to reference
        for child in node.children[1:]:
            # Find overlapping nodes between reference and current child
            ref_global = ref_child.node_indices
            child_global = child.node_indices

            overlap_mask_ref = np.isin(ref_global, child_global)
            overlap_mask_child = np.isin(child_global, ref_global)

            if overlap_mask_ref.sum() > 3:  # Need at least 3 points for Procrustes
                # Get overlapping embeddings
                ref_overlap = ref_embedding[overlap_mask_ref]
                child_overlap = child.embedding[overlap_mask_child]

                # Align using Procrustes
                from scipy.spatial import procrustes
                _, aligned_child, _ = procrustes(ref_overlap, child_overlap)

                # Scale transformation to all child embeddings
                scale = np.linalg.norm(aligned_child) / np.linalg.norm(child_overlap)
                R = aligned_child.T @ child_overlap / (np.linalg.norm(child_overlap) ** 2)

                aligned_embedding = child.embedding @ R * scale
            else:
                # Not enough overlap, use as-is
                aligned_embedding = child.embedding

            # Add to parent embedding
            for local_idx, global_idx in enumerate(child.node_indices):
                parent_idx = np.where(node.node_indices == global_idx)[0][0]
                embedding[parent_idx] += aligned_embedding[local_idx]
                counts[parent_idx] += 1

        # Average overlapping regions
        counts[counts == 0] = 1  # Avoid division by zero
        embedding /= counts[:, np.newaxis]

        return embedding

    def _align_with_aligner(self, node: HierarchicalNode) -> np.ndarray:
        """Align child embeddings using a fresh instance of the configured aligner."""

        # Create a fresh aligner instance to avoid state conflicts
        fresh_aligner = self._create_fresh_aligner()
        
        if self.verbose:
            print(f"    Using {self.aligner_config['type']} alignment for {len(node.children)} children")

        # Create patches from children with proper node mappings
        from ...patch.patches import Patch

        patches = []
        node_to_patch_idx = {}  # Map parent node indices to (patch_idx, local_idx) pairs

        for i, child in enumerate(node.children):
            # Create mapping from child's node indices to parent node indices
            child_nodes_in_parent = []
            for local_idx, global_idx in enumerate(child.node_indices):
                # Find position in parent node
                parent_idx = np.where(node.node_indices == global_idx)[0]
                if len(parent_idx) > 0:
                    parent_node_idx = parent_idx[0]
                    child_nodes_in_parent.append(parent_node_idx)
                    if parent_node_idx not in node_to_patch_idx:
                        node_to_patch_idx[parent_node_idx] = []
                    node_to_patch_idx[parent_node_idx].append((i, local_idx))

            # Create patch with parent node indices
            patch = Patch(
                nodes=np.array(child_nodes_in_parent),
                coordinates=child.embedding
            )
            patches.append(patch)

        # Create patch graph for the aligner
        from ...graphs import TGraph
        from ...patch import create_patches
        
        # Create a simple patch container that the aligner can work with
        patch_container = type('PatchContainer', (), {
            'patches': patches,
            'num_nodes': node.graph.num_nodes
        })()

        # Use the fresh aligner
        fresh_aligner.align_patches(patch_container)

        # Get aligned embedding
        aligned = fresh_aligner.get_aligned_embedding()

        return aligned

    def get_tree_structure(self) -> dict[str, Any]:
        """Get information about the hierarchical tree structure."""

        def get_node_info(node: HierarchicalNode) -> dict[str, Any]:
            info = {
                "level": node.level,
                "num_nodes": node.graph.num_nodes,
                "is_leaf": node.is_leaf,
            }
            if not node.is_leaf:
                info["children"] = [get_node_info(child) for child in node.children]
            return info

        return {
            "max_depth": max(leaf.level for leaf in self.leaves),
            "num_leaves": len(self.leaves),
            "tree": get_node_info(self.root) if self.root else None
        }
