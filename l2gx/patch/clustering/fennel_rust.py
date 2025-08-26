"""
Rust-based Fennel Clustering Implementation

This module provides Python bindings for high-performance Fennel clustering
algorithms implemented in Rust. The Rust implementation offers significant
speedups over the Python/Numba version while maintaining identical functionality.
"""

import numpy as np
import torch
from typing import Optional, Union

# Try to import the Rust implementation
try:
    import l2g_clustering

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    l2g_clustering = None

from l2gx.graphs import TGraph


def fennel_clustering_rust(
    graph: TGraph,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: Optional[float] = None,
    gamma: float = 1.5,
    num_iters: int = 1,
    parallel: bool = False,
    verbose: bool = True,
) -> torch.Tensor:
    """
    High-performance Fennel clustering using Rust implementation

    This function provides a drop-in replacement for the Python/Numba Fennel
    implementation with significant performance improvements.

    Args:
        graph: TGraph object with edge_index and adj_index
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter for utility function (auto-computed if None)
        gamma: Gamma parameter for cluster size penalty (default: 1.5)
        num_iters: Number of clustering iterations (default: 1)
        parallel: Use parallel implementation (default: False)
        verbose: Print progress information (default: True)

    Returns:
        torch.Tensor: Cluster assignment tensor

    Raises:
        ImportError: If Rust implementation is not available
        ValueError: If graph format is incompatible

    Example:
        ```python
        from l2gx.patch.clustering.rust_fennel import fennel_clustering_rust
        from l2gx.graphs import TGraph

        # Create graph
        graph = TGraph(edge_index, num_nodes=1000)

        # Fast Rust clustering
        clusters = fennel_clustering_rust(graph, num_clusters=10)

        # Parallel version for large graphs
        clusters = fennel_clustering_rust(graph, num_clusters=10, parallel=True)
        ```
    """
    if not RUST_AVAILABLE:
        raise ImportError(
            "Rust clustering implementation not available. "
            "Please install the l2g-clustering package or build the Rust extension."
        )

    # Validate inputs
    if not hasattr(graph, "edge_index") or not hasattr(graph, "adj_index"):
        raise ValueError(
            "Graph must be a TGraph with edge_index and adj_index attributes"
        )

    # Convert to numpy arrays for Rust
    edge_index_np = graph.edge_index.cpu().numpy().astype(np.int64)
    adj_index_np = graph.adj_index.cpu().numpy().astype(np.int64)

    # Flatten edge_index for Rust (expects [src0, dst0, src1, dst1, ...])
    edge_index_flat = edge_index_np.flatten()

    # Call appropriate Rust function
    if parallel:
        clusters_np = l2g_clustering.fennel_clustering_parallel_rust(
            edge_index_flat,
            adj_index_np,
            graph.num_nodes,
            num_clusters,
            load_limit,
            alpha,
            gamma,
            num_iters,
            verbose,
        )
    else:
        clusters_np = l2g_clustering.fennel_clustering_rust(
            edge_index_flat,
            adj_index_np,
            graph.num_nodes,
            num_clusters,
            load_limit,
            alpha,
            gamma,
            num_iters,
            verbose,
        )

    # Convert back to PyTorch tensor
    return torch.tensor(clusters_np, dtype=torch.long, device=graph.device)


def fennel_clustering_from_edge_list_rust(
    edge_index: Union[np.ndarray, torch.Tensor],
    num_nodes: int,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: Optional[float] = None,
    gamma: float = 1.5,
    num_iters: int = 1,
    parallel: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """
    Rust Fennel clustering from raw edge list data

    Args:
        edge_index: Edge list as array [2, num_edges] or [num_edges, 2]
        num_nodes: Number of nodes in the graph
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
        parallel: Use parallel implementation (default: False)
        verbose: Whether to print progress info (default: True)

    Returns:
        Cluster assignment array
    """
    if not RUST_AVAILABLE:
        raise ImportError("Rust clustering implementation not available")

    # Convert edge_index to numpy if needed
    if torch.is_tensor(edge_index):
        edge_index = edge_index.cpu().numpy()

    # Ensure correct shape [2, num_edges]
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T

    # Create TGraph to compute adjacency index
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)
    tgraph = TGraph(edge_index_torch, num_nodes=num_nodes)

    # Use the main function
    result_tensor = fennel_clustering_rust(
        tgraph, num_clusters, load_limit, alpha, gamma, num_iters, parallel, verbose
    )

    return result_tensor.cpu().numpy()


def benchmark_rust_vs_python(
    graph: TGraph, num_clusters: int, num_runs: int = 5, **fennel_kwargs
) -> dict:
    """
    Benchmark Rust implementation against Python/Numba version

    Args:
        graph: TGraph to benchmark on
        num_clusters: Number of clusters
        num_runs: Number of benchmark runs (default: 5)
        **fennel_kwargs: Additional arguments for Fennel clustering

    Returns:
        Dictionary with benchmark results
    """
    import time

    results = {
        "rust_times": [],
        "python_times": [],
        "rust_results": [],
        "python_results": [],
    }

    # Benchmark Rust implementation
    if RUST_AVAILABLE:
        for _ in range(num_runs):
            start_time = time.time()
            rust_result = fennel_clustering_rust(
                graph, num_clusters, verbose=False, **fennel_kwargs
            )
            end_time = time.time()
            results["rust_times"].append(end_time - start_time)
            results["rust_results"].append(rust_result)

    # Benchmark Python implementation
    try:
        from .fennel import fennel_clustering

        for _ in range(num_runs):
            start_time = time.time()
            python_result = fennel_clustering(
                graph, num_clusters, verbose=False, **fennel_kwargs
            )
            end_time = time.time()
            results["python_times"].append(end_time - start_time)
            results["python_results"].append(torch.tensor(python_result))

    except ImportError:
        print("Python Fennel implementation not available for comparison")

    # Compute statistics
    if results["rust_times"]:
        results["rust_mean_time"] = np.mean(results["rust_times"])
        results["rust_std_time"] = np.std(results["rust_times"])

    if results["python_times"]:
        results["python_mean_time"] = np.mean(results["python_times"])
        results["python_std_time"] = np.std(results["python_times"])

        if results["rust_times"]:
            results["speedup"] = results["python_mean_time"] / results["rust_mean_time"]

    return results


def is_rust_available() -> bool:
    """Check if Rust implementation is available"""
    return RUST_AVAILABLE


def get_rust_info() -> dict:
    """Get information about the Rust implementation"""
    if not RUST_AVAILABLE:
        return {"available": False, "reason": "l2g_clustering module not found"}

    try:
        # Try to get version info if available
        info = {
            "available": True,
            "module": str(l2g_clustering),
            "functions": ["fennel_clustering_rust", "fennel_clustering_parallel_rust"],
        }

        # Check if functions are callable
        for func_name in info["functions"]:
            if not hasattr(l2g_clustering, func_name):
                info[f"{func_name}_available"] = False
            else:
                info[f"{func_name}_available"] = True

        return info

    except Exception as e:
        return {"available": True, "error": str(e), "module": str(l2g_clustering)}
