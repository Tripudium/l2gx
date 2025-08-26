"""
Graph Clustering Algorithms for Patch Generation

This package provides various graph clustering algorithms used to partition large graphs
into smaller, manageable pieces for the Local2Global algorithm. Each algorithm has
different strengths and is suitable for different types of graphs and use cases.

Available Algorithms:

1. **Fennel Clustering** - Single-pass streaming algorithm with load balancing
2. **Louvain Clustering** - Modularity-based community detection
3. **METIS Clustering** - Multi-level graph partitioning
4. **Spread Clustering** - Degree-based spreading algorithm
5. **Hierarchical Clustering** - Multi-level clustering with size constraints

Usage Examples:
    ```python
    from l2gx.patch.clustering import fennel_clustering, louvain_clustering
    from l2gx.graphs import TGraph

    # Fennel clustering for balanced partitions
    clusters = fennel_clustering(graph, num_clusters=10)

    # Louvain for community-based partitions
    clusters = louvain_clustering(graph)

    # METIS for optimal edge cuts
    clusters = metis_clustering(graph, num_clusters=8)
    ```

For more detailed examples and algorithm-specific options, see the individual
algorithm modules.
"""

# Import all clustering algorithms for backward compatibility
from .fennel import fennel_clustering
from .louvain import (
    louvain_clustering,
    louvain_clustering_with_stats,
    louvain_clustering_multi_resolution,
)
from .metis import (
    metis_clustering,
    metis_clustering_with_stats,
    metis_clustering_weighted,
)
from .spread import (
    spread_clustering,
    spread_clustering_with_stats,
    spread_clustering_balanced,
)
from .hierarchical import (
    hierarchical_clustering,
    hierarchical_agglomerative_clustering,
    adaptive_hierarchical_clustering,
)

# Import Rust implementations if available
try:
    from .fennel_rust import (
        fennel_clustering_rust,
        fennel_clustering_from_edge_list_rust,
        is_rust_available,
        get_rust_info,
        benchmark_rust_vs_python,
    )

    RUST_FENNEL_AVAILABLE = True
except ImportError:
    RUST_FENNEL_AVAILABLE = False

    # Create placeholder functions that explain Rust is not available
    def fennel_clustering_rust(*args, **kwargs):
        raise ImportError(
            "Rust Fennel implementation not available. "
            "Build the Rust extension with: cd rust_clustering && ./build.sh"
        )

    def is_rust_available():
        return False


from .utils import (
    Partition,
    evaluate_clustering,
    compute_edge_cuts,
    compare_clusterings,
    clustering_to_patches,
    convert_graph_format,
    validate_clustering_result,
)

# Import the no-tqdm version for compatibility
try:
    from .fennel import fennel_clustering_no_progress
except ImportError:
    # Fallback if file doesn't exist
    pass

# Also import the legacy _fennel_clustering for backward compatibility
try:
    from .fennel import _fennel_clustering
except ImportError:
    # Create a compatibility wrapper if needed
    def _fennel_clustering(*args, **kwargs):
        """Legacy compatibility wrapper"""
        from .fennel import fennel_clustering_safe

        return fennel_clustering_safe(*args, **kwargs)


# Algorithm registry for easy selection
CLUSTERING_ALGORITHMS = {
    "fennel": fennel_clustering,
    "fennel_rust": fennel_clustering_rust,
    "louvain": louvain_clustering,
    "metis": metis_clustering,
    "spread": spread_clustering,
    "hierarchical": hierarchical_clustering,
}


def get_clustering_algorithm(name):
    """
    Get a clustering algorithm by name

    Args:
        name: Algorithm name ('fennel', 'louvain', 'metis', 'spread', 'hierarchical')

    Returns:
        Clustering function

    Raises:
        ValueError: If algorithm name is not recognized
    """
    if name not in CLUSTERING_ALGORITHMS:
        available = list(CLUSTERING_ALGORITHMS.keys())
        raise ValueError(
            f"Unknown clustering algorithm '{name}'. Available: {available}"
        )

    return CLUSTERING_ALGORITHMS[name]


def list_clustering_algorithms():
    """list all available clustering algorithms"""
    return list(CLUSTERING_ALGORITHMS.keys())


# Export main API
__all__ = [
    # Main clustering functions
    "fennel_clustering",
    "louvain_clustering",
    "metis_clustering",
    "spread_clustering",
    "hierarchical_clustering",
    "hierarchical_agglomerative_clustering",
    "adaptive_hierarchical_clustering",
    # Utility functions
    "Partition",
    "evaluate_clustering",
    "compute_edge_cuts",
    "compare_clusterings",
    "clustering_to_patches",
    "convert_graph_format",
    "validate_clustering_result",
    # Algorithm variants
    "fennel_clustering_safe",
    "louvain_clustering_with_stats",
    "louvain_clustering_multi_resolution",
    "metis_clustering_with_stats",
    "metis_clustering_weighted",
    "spread_clustering_with_stats",
    "spread_clustering_balanced",
    # Rust implementations
    "fennel_clustering_rust",
    "fennel_clustering_from_edge_list_rust",
    "is_rust_available",
    "get_rust_info",
    "benchmark_rust_vs_python",
    "RUST_FENNEL_AVAILABLE",
    # Registry functions
    "get_clustering_algorithm",
    "list_clustering_algorithms",
    "CLUSTERING_ALGORITHMS",
    # Legacy compatibility
    "_fennel_clustering",
    "fennel_clustering_no_progress",
]
