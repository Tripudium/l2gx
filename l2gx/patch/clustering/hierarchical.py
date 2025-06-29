"""
Hierarchical Clustering Algorithms

This module provides hierarchical clustering methods that can create multi-level
cluster structures. These are useful for creating hierarchical patch structures
in the Local2Global algorithm and for handling graphs with multiple scales.

Two main approaches are provided:
1. Recursive hierarchical clustering with size constraints
2. Agglomerative hierarchical clustering with different base methods
"""

import torch
import numpy as np
from collections.abc import Iterable
from typing import Sequence, Callable
from torch_geometric.data import Data
from raphtory import Graph  # pylint: disable=no-name-in-module

from .spread import spread_clustering


def hierarchical_clustering(
    data: Data, m: int, k: int, clustering_function: Callable[[Data, int], torch.Tensor]
) -> list[torch.Tensor]:
    """
    Perform recursive hierarchical clustering on a PyTorch Geometric graph
    
    This function recursively applies a clustering algorithm to create a hierarchy
    of clusters. It splits oversized clusters until all clusters are below the
    size threshold k.
    
    Args:
        data: The input PyTorch Geometric graph
        m: Target number of clusters at each level
        k: Target maximum cluster size
        clustering_function: Function that takes a Data object and number of clusters
                           and returns cluster assignment tensor
                           
    Returns:
        List of cluster assignment tensors for all levels of the hierarchy
        
    Example:
        ```python
        from l2gx.patch.clustering.hierarchical import hierarchical_clustering
        from l2gx.patch.clustering.louvain import louvain_clustering
        
        # Create hierarchical clustering using Louvain as base method
        def louvain_for_data(data, num_clusters):
            # Convert Data to Graph and run Louvain
            # (implementation depends on your graph conversion utilities)
            return louvain_clustering(convert_to_graph(data))
            
        hierarchy = hierarchical_clustering(data, m=4, k=100, louvain_for_data)
        ```
    """
    
    def recursive_clustering(data, m, k):
        """Recursively cluster data until all clusters are small enough"""
        # Apply the clustering function to get initial clusters
        cluster_tensor = clustering_function(data, m)

        # Check the size of each cluster
        unique_clusters, cluster_counts = torch.unique(cluster_tensor, return_counts=True)

        # Store the final cluster assignments
        final_clusters = []

        for cluster_id, count in zip(unique_clusters, cluster_counts):
            # Get nodes in this cluster
            cluster_nodes = (cluster_tensor == cluster_id).nonzero(as_tuple=True)[0]

            if count <= k:
                # If the cluster size is within the limit, keep it as-is
                final_clusters.append((cluster_id.item(), cluster_nodes))
            else:
                # If the cluster size exceeds k, extract subgraph and recurse
                subgraph = extract_subgraph(data, cluster_nodes)
                sub_clusters = recursive_clustering(subgraph, m, k)

                # Adjust sub-cluster IDs to avoid clashes with existing IDs
                max_cluster_id = max(unique_clusters).item() if len(unique_clusters) > 0 else -1
                sub_clusters_adjusted = [
                    (max_cluster_id + 1 + i, cluster_nodes[sub_nodes])
                    for i, (_, sub_nodes) in enumerate(sub_clusters)
                ]
                final_clusters.extend(sub_clusters_adjusted)

        return final_clusters

    def extract_subgraph(data, node_indices):
        """Extract a subgraph containing only the specified nodes"""
        # Create mask for nodes to include
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[node_indices] = True

        # Filter edges to only include those between selected nodes
        edge_mask = mask[data.edge_index[0]] & mask[data.edge_index[1]]
        edge_index = data.edge_index[:, edge_mask]

        # Reindex nodes in the subgraph to be contiguous
        node_mapping = {
            old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)
        }
        
        # Apply remapping to edge indices
        remapped_edges = []
        for src, dst in edge_index.t():
            new_src = node_mapping[src.item()]
            new_dst = node_mapping[dst.item()]
            remapped_edges.append([new_src, new_dst])
        
        if remapped_edges:
            edge_index = torch.tensor(remapped_edges, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create new Data object for subgraph
        subgraph_data = Data(edge_index=edge_index, num_nodes=len(node_indices))
        
        # Copy node features if they exist
        if hasattr(data, 'x') and data.x is not None:
            subgraph_data.x = data.x[node_indices]
            
        return subgraph_data

    # Start recursive clustering from the root level
    return recursive_clustering(data, m, k)


def hierarchical_agglomerative_clustering(
    graph: Graph, 
    method=spread_clustering, 
    levels=None, 
    branch_factors=None
):
    """
    Hierarchical agglomerative clustering using different base methods
    
    This creates a hierarchy by repeatedly clustering graphs and then creating
    coarser graphs from the clusters. Each level uses the specified clustering
    method to partition the current graph.
    
    Args:
        graph: Input graph (Raphtory Graph)
        method: Base clustering method to use at each level (default: spread_clustering)
        levels: Number of hierarchy levels (computed from branch_factors if None)
        branch_factors: Number of clusters at each level (list or single value)
        
    Returns:
        List of cluster assignment tensors, one for each level
        
    Example:
        ```python
        from l2gx.patch.clustering.hierarchical import hierarchical_agglomerative_clustering
        from l2gx.patch.clustering.louvain import louvain_clustering
        
        # 3-level hierarchy with specified branch factors
        hierarchy = hierarchical_agglomerative_clustering(
            graph, 
            method=louvain_clustering,
            branch_factors=[100, 20, 5]
        )
        
        # Automatic levels with uniform branching
        hierarchy = hierarchical_agglomerative_clustering(
            graph,
            levels=3,
            branch_factors=4  # 4-way branching at each level
        )
        ```
    """
    # Set up branch factors
    if branch_factors is None:
        if levels is None:
            levels = 3  # Default to 3 levels
        # Compute uniform branch factors
        branch_factors = [graph.num_nodes ** (1 / (levels + 1)) for _ in range(levels)]
    else:
        if not isinstance(branch_factors, Iterable):
            # Single branch factor - replicate for all levels
            if levels is None:
                levels = 3
            branch_factors = [branch_factors] * levels
        else:
            # List of branch factors
            if levels is None:
                levels = len(branch_factors)
            elif len(branch_factors) != levels:
                raise ValueError(f"levels={levels} does not match {len(branch_factors)=}")
    
    # Compute number of clusters at each level (cumulative product, reversed)
    num_clusters = np.cumprod(branch_factors)[::-1]
    
    clusters = []
    current_graph = graph
    
    for level, n_clusters in enumerate(num_clusters):
        print(f"Level {level}: clustering into {int(n_clusters)} clusters")
        
        # Apply clustering method
        cluster = method(current_graph, int(n_clusters))
        clusters.append(cluster)
        
        # Create coarser graph for next level
        if level < len(num_clusters) - 1:  # Don't create graph for last level
            current_graph = current_graph.partition_graph(cluster)
    
    return clusters


def adaptive_hierarchical_clustering(
    graph: Graph,
    max_cluster_size: int,
    min_cluster_size: int = None,
    method=spread_clustering,
    max_levels: int = 10
):
    """
    Adaptive hierarchical clustering that stops when cluster sizes are appropriate
    
    This variant automatically determines when to stop the hierarchy based on
    cluster size constraints rather than a fixed number of levels.
    
    Args:
        graph: Input graph
        max_cluster_size: Maximum allowed cluster size
        min_cluster_size: Minimum allowed cluster size (default: max_cluster_size // 4)
        method: Base clustering method
        max_levels: Maximum number of levels to prevent infinite recursion
        
    Returns:
        List of cluster assignments and final cluster information
    """
    if min_cluster_size is None:
        min_cluster_size = max(1, max_cluster_size // 4)
    
    clusters = []
    current_graph = graph
    level = 0
    
    while level < max_levels:
        # Estimate number of clusters needed
        estimated_clusters = max(2, current_graph.num_nodes // max_cluster_size)
        
        print(f"Level {level}: {current_graph.num_nodes} nodes, "
              f"targeting {estimated_clusters} clusters")
        
        # Apply clustering
        cluster = method(current_graph, estimated_clusters)
        clusters.append(cluster)
        
        # Check cluster sizes
        cluster_sizes = torch.bincount(cluster)
        max_size = cluster_sizes.max().item()
        min_size = cluster_sizes.min().item()
        
        print(f"  Cluster sizes: [{min_size}, {max_size}], "
              f"target: [{min_cluster_size}, {max_cluster_size}]")
        
        # Stop if all clusters are appropriately sized
        if max_size <= max_cluster_size and min_size >= min_cluster_size:
            print(f"Stopping at level {level}: all clusters appropriately sized")
            break
            
        # Stop if graph is too small to subdivide further
        if current_graph.num_nodes <= max_cluster_size:
            print(f"Stopping at level {level}: graph too small to subdivide")
            break
        
        # Create coarser graph for next iteration
        current_graph = current_graph.partition_graph(cluster)
        level += 1
    
    # Return clusters and summary information
    final_cluster_sizes = torch.bincount(clusters[-1]) if clusters else torch.tensor([])
    
    return {
        'clusters': clusters,
        'levels': level + 1,
        'final_cluster_sizes': final_cluster_sizes,
        'final_max_size': final_cluster_sizes.max().item() if len(final_cluster_sizes) > 0 else 0,
        'final_min_size': final_cluster_sizes.min().item() if len(final_cluster_sizes) > 0 else 0
    }