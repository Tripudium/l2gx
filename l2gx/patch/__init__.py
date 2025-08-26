"""
L2GX Patch Module

This module provides comprehensive patch-based graph processing capabilities for the
Local2Global algorithm, including:

- Patch creation and management
- Graph clustering algorithms
- Graph sparsification techniques
- Lazy coordinate systems for memory efficiency
- Coordinate transformation utilities

Main components:
- patches: Core patch data structures and creation algorithms
- clustering: Graph partitioning algorithms (Fennel, Louvain, METIS, etc.)
- sparsify: Graph sparsification methods (resistance, sampling, k-NN)
- lazy: Memory-efficient lazy loading for large-scale problems
- utils: Coordinate transformation and error computation utilities
"""

# Core patch functionality and high-level generation
from l2gx.patch.patches import (
    # Core classes
    Patch,
    # Core functions
    create_overlapping_patches,
    create_patch_data,
    create_patches,
    # High-level generation functions
    create_patches_by_size,
    rolling_window_graph,
    # Utilities
    list_clustering_methods,
    estimate_patch_parameters,
)

# Clustering algorithms
from l2gx.patch.clustering import (
    hierarchical_clustering,
    fennel_clustering,
    louvain_clustering,
    metis_clustering,
    spread_clustering,
)

# Sparsification methods
from l2gx.patch.sparsify import (
    resistance_sparsify,
    edge_sampling_sparsify,
    nearest_neighbor_sparsify,
    relaxed_spanning_tree,
    hierarchical_sparsify,
    effective_resistances,
    conductance_weighted_graph,
    resistance_weighted_graph,
)

# Utility functions
from l2gx.patch.utils import (
    seed,
    ensure_extension,
    random_gen,
    procrustes_error,
    local_error,
    transform_error,
    orthogonal_mse_error,
    relative_orthogonal_transform,
    nearest_orthogonal,
    relative_scale,
)

__all__ = [
    # Core patch functionality
    "Patch",
    "create_overlapping_patches",
    "create_patch_data",
    "create_patches",
    # High-level patch generation
    "generate_patches",
    "generate_patches_by_size",
    "rolling_window_graph",
    "list_clustering_methods",
    "estimate_patch_parameters",
    # Clustering algorithms
    "hierarchical_clustering",
    "fennel_clustering",
    "louvain_clustering",
    "metis_clustering",
    "spread_clustering",
    # Sparsification methods
    "resistance_sparsify",
    "edge_sampling_sparsify",
    "nearest_neighbor_sparsify",
    "relaxed_spanning_tree",
    "hierarchical_sparsify",
    "effective_resistances",
    "conductance_weighted_graph",
    "resistance_weighted_graph",
    # Utility functions
    "seed",
    "ensure_extension",
    "random_gen",
    "procrustes_error",
    "local_error",
    "transform_error",
    "orthogonal_mse_error",
    "relative_orthogonal_transform",
    "nearest_orthogonal",
    "relative_scale",
]
