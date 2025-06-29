"""
Louvain Clustering Algorithm

The Louvain algorithm is a modularity-based community detection method that's
excellent for finding natural communities in social networks and other graphs
with community structure. It's fast and produces high-quality clusters.

The algorithm works in two phases:
1. Local optimization: Move nodes to maximize modularity
2. Aggregation: Build a new graph of communities

References:
    V. D. Blondel et al. "Fast unfolding of communities in large networks".
    Journal of Statistical Mechanics: Theory and Experiment 2008.10 (2008), P10008.
    DOI: 10.1088/1742-5468/2008/10/P10008
"""

import torch
import community
from raphtory import Graph  # pylint: disable=no-name-in-module


def louvain_clustering(graph: Graph, *args, **kwargs):
    """
    Louvain algorithm for modularity-based community detection
    
    This is a wrapper around the python-louvain package that implements
    the Louvain algorithm for community detection. It finds communities
    by optimizing the modularity measure.
    
    Args:
        graph: Input graph (Raphtory Graph)
        *args: Additional positional arguments passed to community.best_partition
        **kwargs: Additional keyword arguments passed to community.best_partition
        
    Returns:
        torch.Tensor: Cluster assignment tensor
        
    Note:
        This function requires a Raphtory Graph object that can be converted
        to NetworkX format. The algorithm works best on graphs with clear
        community structure.
        
    Example:
        ```python
        from l2gx.patch.clustering.louvain import louvain_clustering
        
        # Basic community detection
        clusters = louvain_clustering(graph)
        
        # With custom resolution parameter
        clusters = louvain_clustering(graph, resolution=1.2)
        ```
    """
    # Convert to NetworkX and ensure undirected for community detection
    nx_graph = graph.to_networkx().to_undirected()
    
    # Run Louvain community detection
    clusters = community.best_partition(nx_graph, *args, **kwargs)
    
    # Convert back to tensor format
    cluster_tensor = torch.tensor(
        [clusters[i] for i in range(graph.num_nodes)], 
        dtype=torch.long
    )
    
    return cluster_tensor


def louvain_clustering_with_stats(graph: Graph, verbose=True, *args, **kwargs):
    """
    Louvain clustering with additional statistics reporting
    
    Args:
        graph: Input graph (Raphtory Graph)
        verbose: Whether to print statistics (default: True)
        *args: Additional positional arguments passed to community.best_partition
        **kwargs: Additional keyword arguments passed to community.best_partition
        
    Returns:
        dict: Dictionary containing:
            - 'clusters': Cluster assignment tensor
            - 'modularity': Modularity score of the clustering
            - 'num_communities': Number of communities found
            - 'community_sizes': Sizes of each community
    """
    # Convert to NetworkX
    nx_graph = graph.to_networkx().to_undirected()
    
    # Run Louvain clustering
    partition = community.best_partition(nx_graph, *args, **kwargs)
    
    # Calculate modularity
    modularity = community.modularity(partition, nx_graph)
    
    # Convert to tensor
    cluster_tensor = torch.tensor(
        [partition[i] for i in range(graph.num_nodes)], 
        dtype=torch.long
    )
    
    # Calculate statistics
    unique_communities = torch.unique(cluster_tensor)
    num_communities = len(unique_communities)
    community_sizes = torch.bincount(cluster_tensor)
    
    if verbose:
        print(f"Louvain clustering results:")
        print(f"  Communities found: {num_communities}")
        print(f"  Modularity: {modularity:.4f}")
        print(f"  Community sizes: {community_sizes.tolist()}")
        print(f"  Average community size: {community_sizes.float().mean():.1f}")
    
    return {
        'clusters': cluster_tensor,
        'modularity': modularity,
        'num_communities': num_communities,
        'community_sizes': community_sizes
    }


def louvain_clustering_multi_resolution(graph: Graph, resolutions=None, verbose=True):
    """
    Run Louvain clustering with multiple resolution parameters
    
    The resolution parameter controls the size of communities. Higher values
    lead to smaller communities, lower values lead to larger communities.
    
    Args:
        graph: Input graph (Raphtory Graph)
        resolutions: List of resolution values to try (default: [0.5, 1.0, 1.5, 2.0])
        verbose: Whether to print results for each resolution
        
    Returns:
        dict: Results for each resolution parameter
    """
    if resolutions is None:
        resolutions = [0.5, 1.0, 1.5, 2.0]
    
    results = {}
    
    for resolution in resolutions:
        if verbose:
            print(f"\nTrying resolution = {resolution}")
            
        result = louvain_clustering_with_stats(
            graph, verbose=verbose, resolution=resolution
        )
        results[resolution] = result
    
    if verbose:
        print(f"\nSummary across resolutions:")
        for res in resolutions:
            r = results[res]
            print(f"  Resolution {res}: {r['num_communities']} communities, "
                  f"modularity = {r['modularity']:.4f}")
    
    return results