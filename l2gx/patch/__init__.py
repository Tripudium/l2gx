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

# Core patch functionality
from .patches import (
    create_overlapping_patches,
    create_patch_data,
    Patch,
    MeanAggregatorPatch,
    FilePatch,
)

# Lazy coordinate systems
from .lazy import (
    BaseLazyCoordinates,
    LazyMeanAggregatorCoordinates,
    LazyFileCoordinates,
)

# Clustering algorithms
from .clustering import (
    hierarchical_clustering,
    fennel_clustering,
    louvain_clustering,
    metis_clustering,
    spread_clustering,
)

# Sparsification methods
from .sparsify import (
    resistance_sparsify,
    edge_sampling_sparsify,
    nearest_neighbor_sparsify,
    relaxed_spanning_tree,
    hierarchical_sparsify,
    effective_resistances,
    conductance_weighted_graph,
    resistance_weighted_graph,
)

# High-level patch generation functions
from .generate import (
    generate_patches,
    generate_patches_by_size,
    generate_patches_adaptive,
    list_clustering_methods,
    estimate_patch_parameters,
)

# Utility functions
from .utils import (
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
    "create_overlapping_patches",
    "create_patch_data",
    "Patch",
    "MeanAggregatorPatch",
    "FilePatch",
    
    # Lazy coordinate systems
    "BaseLazyCoordinates",
    "LazyMeanAggregatorCoordinates", 
    "LazyFileCoordinates",
    
    # High-level patch generation
    "generate_patches",
    "generate_patches_by_size",
    "generate_patches_adaptive",
    "list_clustering_methods",
    "estimate_patch_parameters",
    
    # Clustering algorithms
    "hierarchical_clustering",
    "fennel_clustering",
    "louvain_clustering",
    "metis_clustering",
    "spread_clustering",
    "hierarchical_aglomerative_clustering",
    
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
