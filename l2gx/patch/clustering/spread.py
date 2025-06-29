"""
Spread Clustering Algorithm

Spread clustering is a degree-based spreading algorithm that's simple and fast
for basic partitioning needs. It works by selecting seed nodes (typically high-degree
nodes) and then spreading from these seeds to create balanced clusters.

The algorithm:
1. Select seed nodes (high-degree or random)
2. Spread from seeds to neighbors based on weighted adjacency
3. Continue until all nodes are assigned

This is a good baseline method that's computationally efficient and works well
for graphs where degree-based spreading makes sense.
"""

import torch
from raphtory import Graph  # pylint: disable=no-name-in-module


def spread_clustering(graph: Graph, num_clusters: int, max_degree_init=True):
    """
    Degree-based spread clustering algorithm
    
    This algorithm creates clusters by spreading from seed nodes. It starts with
    high-degree nodes as seeds and spreads to their neighbors based on weighted
    adjacency, creating balanced clusters.
    
    Args:
        graph: Input graph (Raphtory Graph)
        num_clusters: Target number of clusters
        max_degree_init: If True, use highest degree nodes as seeds;
                        if False, use random weighted selection (default: True)
                        
    Returns:
        torch.Tensor: Cluster assignment tensor
        
    Note:
        The algorithm may create more clusters than requested if the graph
        has disconnected components. This is handled automatically.
        
    Example:
        ```python
        from l2gx.patch.clustering.spread import spread_clustering
        
        # Use highest degree nodes as seeds
        clusters = spread_clustering(graph, num_clusters=10)
        
        # Use random weighted selection for seeds
        clusters = spread_clustering(graph, num_clusters=10, max_degree_init=False)
        ```
    """
    # Initialize cluster assignments
    clusters = torch.full(
        (graph.num_nodes,), -1, dtype=torch.long, device=graph.device
    )
    
    # Select seed nodes
    if max_degree_init:
        # Use highest degree nodes as seeds
        seeds = torch.topk(torch.as_tensor(graph.degree), k=num_clusters).indices
    else:
        # Use random weighted selection based on degree
        seeds = torch.multinomial(
            torch.as_tensor(graph.degree), num_clusters, replacement=False
        )

    # Assign seeds to clusters
    clusters[seeds] = torch.arange(num_clusters)
    
    # Initialize spread weights
    spread_weights = torch.zeros(
        (num_clusters, graph.num_nodes), dtype=torch.double, device=graph.device
    )
    spread_weights[:, seeds] = -1  # Mark seeds as assigned
    
    # Track unassigned nodes
    unassigned = clusters < 0
    
    # Initialize spread weights from seeds
    for seed in seeds:
        c = clusters[seed]
        inds, weights = graph.adj_weighted(seed)
        keep = unassigned[inds]
        spread_weights[c, inds[keep]] += weights[keep] / graph.strength[inds[keep]]

    num_unassigned = graph.num_nodes - num_clusters

    # Spread clustering loop
    while num_unassigned > 0:
        progress = False
        
        for c in range(num_clusters):
            # Find node with highest spread weight for this cluster
            node = torch.argmax(spread_weights[c])
            
            if spread_weights[c, node] > 0:
                progress = True
                
                # Assign node to cluster
                clusters[node] = c
                spread_weights[:, node] = -1  # Mark as assigned
                unassigned[node] = False
                num_unassigned -= 1
                
                # Update spread weights from newly assigned node
                inds, weights = graph.adj_weighted(node)
                keep = unassigned[inds]
                spread_weights[c, inds[keep]] += (
                    weights[keep] / graph.strength[inds[keep]]
                )
        
        # Handle disconnected components
        if not progress:
            print("Increasing number of clusters due to disconnected components")
            unassigned_nodes = torch.nonzero(unassigned).ravel()
            
            if max_degree_init:
                # Select highest degree unassigned node
                seed = unassigned_nodes[
                    torch.argmax(torch.as_tensor(graph.degree[unassigned_nodes]))
                ]
            else:
                # Random weighted selection from unassigned nodes
                seed = unassigned_nodes[
                    torch.multinomial(
                        torch.as_tensor(graph.degree[unassigned_nodes]), 1
                    )
                ]
            
            # Create new cluster
            clusters[seed] = num_clusters
            spread_weights = torch.cat(
                (
                    spread_weights,
                    torch.zeros(
                        (1, graph.num_nodes), dtype=torch.double, device=graph.device
                    ),
                )
            )
            unassigned[seed] = False
            spread_weights[:, seed] = -1
            
            # Initialize spread from new seed
            inds, weights = graph.adj_weighted(seed)
            keep = unassigned[inds]
            spread_weights[num_clusters, inds[keep]] += (
                weights[keep] / graph.strength[inds[keep]]
            )
            
            num_clusters += 1
            num_unassigned -= 1
    
    return clusters


def spread_clustering_with_stats(graph: Graph, num_clusters: int, max_degree_init=True, verbose=True):
    """
    Spread clustering with additional statistics reporting
    
    Args:
        graph: Input graph (Raphtory Graph)
        num_clusters: Target number of clusters
        max_degree_init: Whether to use highest degree nodes as seeds
        verbose: Whether to print statistics
        
    Returns:
        dict: Dictionary containing:
            - 'clusters': Cluster assignment tensor
            - 'num_clusters': Actual number of clusters created
            - 'cluster_sizes': Sizes of each cluster
            - 'disconnected_components': Number of additional clusters from disconnected components
    """
    initial_clusters = num_clusters
    clusters = spread_clustering(graph, num_clusters, max_degree_init)
    
    # Calculate statistics
    actual_num_clusters = len(torch.unique(clusters))
    cluster_sizes = torch.bincount(clusters)
    disconnected_components = actual_num_clusters - initial_clusters
    
    if verbose:
        print(f"Spread clustering results:")
        print(f"  Requested clusters: {initial_clusters}")
        print(f"  Actual clusters: {actual_num_clusters}")
        print(f"  Disconnected components: {disconnected_components}")
        print(f"  Cluster sizes: {cluster_sizes.tolist()}")
        print(f"  Average cluster size: {cluster_sizes.float().mean():.1f}")
        print(f"  Size std deviation: {cluster_sizes.float().std():.1f}")
    
    return {
        'clusters': clusters,
        'num_clusters': actual_num_clusters,
        'cluster_sizes': cluster_sizes,
        'disconnected_components': disconnected_components
    }


def spread_clustering_balanced(graph: Graph, num_clusters: int, max_imbalance=1.2, verbose=True):
    """
    Spread clustering with better load balancing
    
    This variant of spread clustering tries to maintain more balanced cluster sizes
    by penalizing clusters that are already large.
    
    Args:
        graph: Input graph (Raphtory Graph)
        num_clusters: Target number of clusters
        max_imbalance: Maximum allowed imbalance ratio (default: 1.2)
        verbose: Whether to print statistics
        
    Returns:
        torch.Tensor: Cluster assignment tensor
    """
    target_size = graph.num_nodes / num_clusters
    max_size = int(target_size * max_imbalance)
    
    # Start with regular spread clustering
    clusters = spread_clustering(graph, num_clusters, max_degree_init=True)
    
    # Check for imbalanced clusters
    cluster_sizes = torch.bincount(clusters)
    oversized = cluster_sizes > max_size
    
    if oversized.any() and verbose:
        print(f"Rebalancing {oversized.sum()} oversized clusters...")
        
        # Simple rebalancing: move nodes from oversized clusters to undersized ones
        # This is a basic implementation - more sophisticated methods could be used
        undersized = cluster_sizes < target_size * 0.8
        
        if undersized.any():
            oversized_clusters = torch.nonzero(oversized).flatten()
            undersized_clusters = torch.nonzero(undersized).flatten()
            
            for over_c in oversized_clusters:
                over_nodes = torch.nonzero(clusters == over_c).flatten()
                excess = cluster_sizes[over_c] - max_size
                
                if excess > 0 and len(undersized_clusters) > 0:
                    # Move some nodes to undersized clusters
                    nodes_to_move = over_nodes[:excess]
                    for i, node in enumerate(nodes_to_move):
                        target_cluster = undersized_clusters[i % len(undersized_clusters)]
                        clusters[node] = target_cluster
    
    final_sizes = torch.bincount(clusters)
    if verbose:
        print(f"Balanced spread clustering:")
        print(f"  Final cluster sizes: {final_sizes.tolist()}")
        print(f"  Size range: [{final_sizes.min()}, {final_sizes.max()}]")
        print(f"  Imbalance ratio: {final_sizes.max().float() / final_sizes.mean():.2f}")
    
    return clusters