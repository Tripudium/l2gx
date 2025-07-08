"""
High-Level Patch Generation Functions

This module provides streamlined functions for generating patches from graphs,
combining clustering, sparsification, and patch creation into easy-to-use interfaces.
"""

import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple

from l2gx.graphs import TGraph
from l2gx.patch.patches import create_patch_data, Patch
from l2gx.patch.clustering import (
    get_clustering_algorithm, CLUSTERING_ALGORITHMS
)

# Try to import Rust implementations if available
try:
    from l2gx.patch.clustering import fennel_clustering_rust, is_rust_available
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def generate_patches(
    graph: TGraph,
    patch_size: Optional[int] = None,
    num_patches: Optional[int] = None,
    clustering_method: str = "metis",
    min_overlap: Optional[int] = None,
    target_overlap: Optional[int] = None,
    sparsify_method: str = "resistance",
    target_patch_degree: int = 4,
    clustering_params: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> Tuple[List[Patch], torch.Tensor]:
    """
    Generate patches from a graph with specified parameters
    
    This is the main high-level function for patch generation. It combines
    clustering, sparsification, and patch creation into a single streamlined workflow.
    
    Args:
        graph: Input graph (TGraph)
        patch_size: Target nodes per patch (exclusive with num_patches)
        num_patches: Target number of patches (exclusive with patch_size)
        clustering_method: Clustering algorithm to use
            Options: 'fennel', 'louvain', 'metis', 'spread', 'hierarchical'
        min_overlap: Minimum overlap between patches (default: computed from patch_size)
        target_overlap: Target overlap between patches (default: computed from patch_size)
        sparsify_method: Graph sparsification method
            Options: 'resistance', 'rmst', 'none', 'edge_sampling', 'knn'
        target_patch_degree: Target degree for patch graph sparsification (default: 4)
        clustering_params: Additional parameters for clustering algorithm
        verbose: Print progress information
        
    Returns:
        Tuple of (patches, patch_graph) where:
        - patches: List of Patch objects
        - patch_graph: Tensor representing patch adjacency
        
    Example:
        ```python
        from l2gx.graphs import TGraph
        from l2gx.patch import generate_patches
        
        # Create graph
        graph = TGraph(edge_index, num_nodes=10000)
        
        # Generate patches by target size
        patches, patch_graph = generate_patches(
            graph, 
            patch_size=100,
            clustering_method='fennel',
            min_overlap=20
        )
        
        # Generate patches by number
        patches, patch_graph = generate_patches(
            graph,
            num_patches=50,
            clustering_method='louvain'
        )
        ```
    """
    if verbose:
        print(f"Generating patches from graph with {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Validate input parameters
    if patch_size is not None and num_patches is not None:
        raise ValueError("Cannot specify both patch_size and num_patches")
    
    if patch_size is None and num_patches is None:
        raise ValueError("Must specify either patch_size or num_patches")
    
    # Calculate parameters
    if patch_size is not None:
        num_patches = max(1, graph.num_nodes // patch_size)
        if verbose:
            print(f"Target patch size: {patch_size} → {num_patches} patches")
    else:
        patch_size = max(1, graph.num_nodes // num_patches)
        if verbose:
            print(f"Target patches: {num_patches} → ~{patch_size} nodes per patch")
    
    # Set overlap defaults if not specified
    if min_overlap is None:
        min_overlap = max(1, patch_size // 10)  # 10% of patch size
    
    if target_overlap is None:
        target_overlap = max(min_overlap, patch_size // 5)  # 20% of patch size
    
    if verbose:
        print(f"Overlap: min={min_overlap}, target={target_overlap}")
    
    # Set up clustering parameters
    clustering_params = clustering_params or {}
    clustering_params.setdefault('verbose', verbose)
    
    # Step 1: Cluster the graph
    if verbose:
        print(f"Step 1: Clustering with {clustering_method}...")
    
    clusters = run_clustering(
        graph, num_patches, clustering_method, clustering_params
    )
    
    if verbose:
        unique_clusters = len(torch.unique(clusters[clusters >= 0]))
        cluster_sizes = torch.bincount(clusters[clusters >= 0])
        print(f"Clustering complete: {unique_clusters} clusters, "
              f"sizes: [{cluster_sizes.min()}, {cluster_sizes.max()}]")
    
    # Step 2: Create patches with overlap and sparsification
    if verbose:
        print(f"Step 2: Creating patches with {sparsify_method} sparsification...")
    
    patches, patch_graph = create_patch_data(
        graph=graph,
        partition_tensor=clusters,
        min_overlap=min_overlap,
        target_overlap=target_overlap,
        sparsify_method=sparsify_method,
        target_patch_degree=target_patch_degree,
        verbose=verbose
    )
    
    if verbose:
        patch_sizes = [len(patch.nodes) for patch in patches]
        print(f"Patch generation complete: {len(patches)} patches created")
        print(f"Patch sizes: [{min(patch_sizes)}, {max(patch_sizes)}], "
              f"avg: {np.mean(patch_sizes):.1f}")
    
    return patches, patch_graph


def run_clustering(
    graph: TGraph,
    num_clusters: int,
    method: str,
    use_rust: bool = True,
    params: Optional[Dict[str, Any]] = None
) -> torch.Tensor:
    """
    Run clustering algorithm on graph
    
    Args:
        graph: Input graph
        num_clusters: Target number of clusters
        method: Clustering method name
        use_rust: Use Rust implementation if available
        params: Additional clustering parameters
        
    Returns:
        Cluster assignment tensor
    """
    params = params or {}
    
    # Handle Rust Fennel specially
    if method == "fennel" and use_rust and RUST_AVAILABLE:
        try:
            return fennel_clustering_rust(
                graph, num_clusters, **params
            )
        except Exception as e:
            print(f"Rust Fennel failed ({e}), falling back to Python")
            method = "fennel"  # Fall back to Python
    
    # Get clustering function
    if method in CLUSTERING_ALGORITHMS:
        clustering_func = CLUSTERING_ALGORITHMS[method]
    else:
        try:
            clustering_func = get_clustering_algorithm(method)
        except ValueError:
            available = list(CLUSTERING_ALGORITHMS.keys())
            raise ValueError(f"Unknown clustering method '{method}'. Available: {available}")
    
    # Run clustering
    if method == "fennel":
        # Convert TGraph to Raphtory format for Fennel
        try:
            raphtory_graph = graph.to_raphtory()
            return clustering_func(raphtory_graph, num_clusters, **params)
        except Exception:
            # Fallback: use safe implementation
            from l2gx.patch.clustering.fennel import fennel_clustering_safe
            edge_index_np = graph.edge_index.cpu().numpy()
            adj_index_np = graph.adj_index.cpu().numpy()
            clusters_np = fennel_clustering_safe(
                edge_index_np, adj_index_np, graph.num_nodes, num_clusters, **params
            )
            return torch.tensor(clusters_np, dtype=torch.long, device=graph.device)
    
    elif method == "metis":
        # METIS requires TGraph format
        return clustering_func(graph, num_clusters, **params)
    
    elif method in ["louvain", "spread"]:
        # These require Raphtory format
        raphtory_graph = graph.to_raphtory()
        return clustering_func(raphtory_graph, num_clusters, **params)
    
    else:
        # Try the function directly
        return clustering_func(graph, num_clusters, **params)


def generate_patches_by_size(
    graph: TGraph,
    target_patch_size: int,
    size_tolerance: float = 0.2,
    max_iterations: int = 3,
    **kwargs
) -> Tuple[List[Patch], torch.Tensor]:
    """
    Generate patches with target size, adjusting clustering until size constraints are met
    
    Args:
        graph: Input graph
        target_patch_size: Desired patch size
        size_tolerance: Acceptable size deviation (default: 20%)
        max_iterations: Maximum attempts to achieve target size
        **kwargs: Additional arguments passed to generate_patches
        
    Returns:
        Tuple of (patches, patch_graph)
    """
    min_size = int(target_patch_size * (1 - size_tolerance))
    max_size = int(target_patch_size * (1 + size_tolerance))
    
    for iteration in range(max_iterations):
        num_patches = max(1, graph.num_nodes // target_patch_size)
        
        patches, patch_graph = generate_patches(
            graph, num_patches=num_patches, **kwargs
        )
        
        patch_sizes = [len(patch.nodes) for patch in patches]
        avg_size = np.mean(patch_sizes)
        
        if min_size <= avg_size <= max_size:
            if kwargs.get('verbose', True):
                print(f"Target size achieved in {iteration + 1} iterations: "
                      f"avg={avg_size:.1f} (target={target_patch_size})")
            return patches, patch_graph
        
        # Adjust target for next iteration
        if avg_size < min_size:
            target_patch_size = int(target_patch_size * 0.9)  # Smaller patches
        elif avg_size > max_size:
            target_patch_size = int(target_patch_size * 1.1)  # Larger patches
        
        if kwargs.get('verbose', True):
            print(f"Iteration {iteration + 1}: avg_size={avg_size:.1f}, "
                  f"adjusting target to {target_patch_size}")
    
    # Return best attempt
    if kwargs.get('verbose', True):
        print(f"Could not achieve target size in {max_iterations} iterations, "
              f"returning best attempt (avg={np.mean(patch_sizes):.1f})")
    
    return patches, patch_graph


def generate_patches_adaptive(
    graph: TGraph,
    max_patch_size: int,
    min_patch_size: Optional[int] = None,
    clustering_method: str = "fennel",
    **kwargs
) -> Tuple[List[Patch], torch.Tensor]:
    """
    Adaptive patch generation that automatically determines optimal parameters
    
    Args:
        graph: Input graph
        max_patch_size: Maximum allowed patch size
        min_patch_size: Minimum allowed patch size (default: max_patch_size // 4)
        clustering_method: Clustering algorithm to use
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (patches, patch_graph)
    """
    if min_patch_size is None:
        min_patch_size = max(1, max_patch_size // 4)
    
    # Estimate optimal number of patches
    target_size = (max_patch_size + min_patch_size) // 2
    num_patches = max(1, graph.num_nodes // target_size)
    
    # Adaptive overlap based on graph density
    density = graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1) / 2)
    base_overlap = max(1, target_size // 10)
    
    if density > 0.1:  # Dense graph
        target_overlap = base_overlap * 2
    elif density < 0.01:  # Sparse graph
        target_overlap = base_overlap // 2
    else:
        target_overlap = base_overlap
    
    kwargs.setdefault('min_overlap', base_overlap)
    kwargs.setdefault('target_overlap', target_overlap)
    
    if kwargs.get('verbose', True):
        print(f"Adaptive parameters: density={density:.4f}, "
              f"target_size={target_size}, overlap={target_overlap}")
    
    return generate_patches(
        graph,
        num_patches=num_patches,
        clustering_method=clustering_method,
        **kwargs
    )


def list_clustering_methods() -> Dict[str, str]:
    """
    List available clustering methods with descriptions
    
    Returns:
        Dictionary mapping method names to descriptions
    """
    descriptions = {
        'fennel': 'Single-pass streaming algorithm with load balancing',
        'louvain': 'Modularity-based community detection',
        'metis': 'Multi-level graph partitioning (optimal edge cuts)',
        'spread': 'Degree-based spreading algorithm',
        'hierarchical': 'Multi-level clustering with size constraints'
    }
    
    if RUST_AVAILABLE:
        descriptions['fennel'] += ' (Rust accelerated)'
    
    return descriptions


def estimate_patch_parameters(
    graph: TGraph,
    target_patch_size: Optional[int] = None,
    target_num_patches: Optional[int] = None
) -> Dict[str, int]:
    """
    Estimate reasonable patch generation parameters for a graph
    
    Args:
        graph: Input graph
        target_patch_size: Desired patch size (optional)
        target_num_patches: Desired number of patches (optional)
        
    Returns:
        Dictionary with recommended parameters
    """
    if target_patch_size is not None:
        num_patches = max(1, graph.num_nodes // target_patch_size)
        patch_size = target_patch_size
    elif target_num_patches is not None:
        num_patches = target_num_patches
        patch_size = max(1, graph.num_nodes // num_patches)
    else:
        # Auto-estimate based on graph size
        if graph.num_nodes < 1000:
            num_patches = max(2, graph.num_nodes // 100)
        elif graph.num_nodes < 10000:
            num_patches = max(5, graph.num_nodes // 200)
        else:
            num_patches = max(10, graph.num_nodes // 500)
        
        patch_size = graph.num_nodes // num_patches
    
    # Estimate overlaps
    min_overlap = max(1, patch_size // 10)
    target_overlap = max(min_overlap, patch_size // 5)
    
    # Recommend clustering method based on graph size
    if graph.num_nodes > 10000:
        clustering_method = "fennel"  # Good for large graphs
    elif graph.num_nodes > 1000:
        clustering_method = "metis"   # Good balance of quality and speed
    else:
        clustering_method = "louvain" # Good for small graphs
    
    return {
        'num_patches': num_patches,
        'patch_size': patch_size,
        'min_overlap': min_overlap,
        'target_overlap': target_overlap,
        'clustering_method': clustering_method
    }