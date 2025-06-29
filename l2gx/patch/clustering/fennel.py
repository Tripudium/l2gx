"""
Fennel Clustering Algorithm

FENNEL (Fast Efficient Network partitioning) is a single-pass streaming 
graph clustering algorithm that balances cluster sizes while minimizing edge cuts.
It's particularly well-suited for large graphs that don't fit in memory.

The algorithm processes nodes in streaming fashion, assigning each node to the
cluster that maximizes a utility function combining:
1. Number of neighbors already in the cluster (modularity)
2. Penalty for cluster size imbalance (load balancing)

References:
    C. Tsourakakis et al. "FENNEL: Streaming Graph Partitioning for Massive Scale Graphs".
    WSDM'14 (2014) doi: 10.1145/2556195.2556213
"""

import numpy as np
import numba
import torch
from raphtory import Graph  # pylint: disable=no-name-in-module
from tqdm import tqdm

from l2gx.utils import tqdm_close


def fennel_clustering(
    graph: Graph,
    num_clusters,
    load_limit=1.1,
    alpha=None,
    gamma=1.5,
    num_iters=1,
    clusters=None,
):
    """
    FENNEL single-pass graph clustering algorithm
    
    Args:
        graph: Input graph (Raphtory Graph)
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter for utility function (auto-computed if None)
        gamma: Gamma parameter for cluster size penalty (default: 1.5)
        num_iters: Number of clustering iterations (default: 1)
        clusters: Initial clustering to refine (optional)
        
    Returns:
        torch.Tensor: Cluster assignment tensor
        
    Note:
        This function requires a Raphtory Graph object. For other graph formats,
        use the utility functions in fennel_utils.py
    """
    if clusters is None:
        clusters = _fennel_clustering(
            graph.edge_index,
            graph.adj_index,
            graph.num_nodes,
            num_clusters,
            load_limit,
            alpha,
            gamma,
            num_iters,
        )
    else:
        clusters = _fennel_clustering(
            graph.edge_index,
            graph.adj_index,
            graph.num_nodes,
            num_clusters,
            load_limit,
            alpha,
            gamma,
            num_iters,
            clusters,
        )
    return torch.as_tensor(clusters)


@numba.njit
def _fennel_clustering(
    edge_index: np.ndarray,
    adj_index: np.ndarray,
    num_nodes: int,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float | None = None,
    gamma: float = 1.5,
    num_iters: int = 1,
    clusters=np.empty(0, dtype=np.int64),
):
    """
    Core FENNEL clustering implementation with Numba JIT compilation
    
    This is the performance-critical inner loop of the FENNEL algorithm,
    compiled with Numba for maximum speed.
    
    Args:
        edge_index: Edge list array [2, num_edges]
        adj_index: Adjacency index for fast neighbor lookup
        num_nodes: Number of nodes in the graph
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor
        alpha: Utility function parameter
        gamma: Cluster size penalty parameter
        num_iters: Number of iterations
        clusters: Initial clustering (optional)
        
    Returns:
        np.ndarray: Cluster assignments
    """
    num_edges = edge_index.shape[1]

    if alpha is None:
        alpha = num_edges * (num_clusters ** (gamma - 1)) / (num_nodes**gamma)

    partition_sizes = np.zeros(num_clusters, dtype=np.int64)
    if clusters.size == 0:
        clusters = np.full((num_nodes,), -1, dtype=np.int64)
    else:
        # There is already a clustering, so we need to copy it and update the partition sizes
        clusters = np.copy(clusters)
        for index in clusters:
            partition_sizes[index] += 1

    # Maximum number of nodes per cluster
    load_limit *= num_nodes / num_clusters

    assert alpha
    deltas = -alpha * gamma * (partition_sizes ** (gamma - 1))

    with numba.objmode:
        progress = tqdm(total=num_nodes)

    for it in range(num_iters):
        not_converged = 0

        progress_it = 0
        for i in range(num_nodes):
            cluster_indices = np.empty(
                (adj_index[i + 1] - adj_index[i],), dtype=np.int64
            )
            for ni, index in enumerate(range(adj_index[i], adj_index[i + 1])):
                cluster_indices[ni] = clusters[edge_index[1, index]]
            old_cluster = clusters[i]
            if old_cluster >= 0:
                partition_sizes[old_cluster] -= 1
            cluster_indices = cluster_indices[cluster_indices >= 0]

            if cluster_indices.size > 0:
                c_size = np.zeros(num_clusters, dtype=np.int64)
                for index in cluster_indices:
                    c_size[index] += 1
                ind = np.argmax(deltas + c_size)
            else:
                ind = np.argmax(deltas)
                
            clusters[i] = ind
            partition_sizes[ind] += 1
            if partition_sizes[ind] == load_limit:
                deltas[ind] = -np.inf
            else:
                deltas[ind] = -alpha * gamma * (partition_sizes[ind] ** (gamma - 1))
            not_converged += ind != old_cluster

            if i % 10000 == 0 and i > 0:
                progress_it = i
                with numba.objmode:
                    progress.update(10000)
                    
        with numba.objmode:
            progress.update(num_nodes - progress_it)

        print(f"iteration: {str(it)}, not converged: {str(not_converged)}")

        if not_converged == 0:
            print(f"converged after {str(it)} iterations.")
            break
            
    with numba.objmode:
        tqdm_close(progress)

    return clusters


def fennel_clustering_safe(
    edge_index: np.ndarray,
    adj_index: np.ndarray, 
    num_nodes: int,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1,
    verbose: bool = True
) -> np.ndarray:
    """
    Safe wrapper for Fennel clustering that avoids tqdm/Numba issues.
    
    This function provides an alternative to the main fennel_clustering
    that doesn't use progress bars inside Numba-compiled code, avoiding
    potential compilation issues.
    
    Args:
        edge_index: Edge index array [2, num_edges]
        adj_index: Adjacency index for fast neighbor lookup
        num_nodes: Number of nodes in the graph
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
        verbose: Whether to print progress info (default: True)
    
    Returns:
        Cluster assignment array
    """
    if verbose:
        print(f"Starting Fennel clustering: {num_nodes} nodes → {num_clusters} clusters")
    
    # Use the no-progress version to avoid Numba issues
    from .fennel_no_tqdm import fennel_clustering_no_progress
    
    clusters = fennel_clustering_no_progress(
        edge_index=edge_index,
        adj_index=adj_index,
        num_nodes=num_nodes,
        num_clusters=num_clusters,
        load_limit=load_limit,
        alpha=alpha,
        gamma=gamma,
        num_iters=num_iters
    )
    
    if verbose:
        unique_clusters = len(np.unique(clusters[clusters >= 0]))
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        print(f"Fennel completed: {unique_clusters} clusters, sizes: {cluster_sizes}")
    
    return clusters