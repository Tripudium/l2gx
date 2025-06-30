"""
Fennel clustering implementation without tqdm progress bars to avoid Numba issues.
"""

import numpy as np
import numba


@numba.njit
def _fennel_clustering(
    edge_index: np.ndarray,
    adj_index: np.ndarray,
    num_nodes: int,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1,
    clusters=np.empty(0, dtype=np.int64),
):
    """
    FENNEL single-pass graph clustering algorithm
    
    Args:
        edge_index: Edge index array [2, num_edges]
        adj_index: Adjacency index for fast neighbor lookup
        num_nodes: Number of nodes in the graph
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
        clusters: Input clustering to refine (optional)
    
    Returns:
        Cluster assignment array
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

    for it in range(num_iters):
        not_converged = 0

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

        # Print progress info (but not inside progress bars)
        print(f"Fennel iteration: {it}, not converged: {not_converged}")

        if not_converged == 0:
            print(f"Fennel converged after {it} iterations.")
            break

    return clusters


def fennel_clustering(
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
    Safe wrapper for Fennel clustering that handles progress reporting.
    
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
    
    # Run the core algorithm without progress bars
    clusters = _fennel_clustering(
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