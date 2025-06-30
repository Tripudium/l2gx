"""
METIS Clustering Algorithm

METIS is a multi-level graph partitioning algorithm that's optimal for creating
balanced partitions with minimal edge cuts. It's the industry standard for
graph partitioning and is particularly effective for finite element meshes
and sparse matrices.

The algorithm works in three phases:
1. Coarsening: Reduce graph size by merging vertices
2. Partitioning: Partition the coarsest graph
3. Refinement: Project back and refine at each level

References:
    "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs".
    George Karypis and Vipin Kumar.
    SIAM Journal on Scientific Computing, Vol. 20, No. 1, pp. 359-392, 1999.
"""

import torch
import pymetis
from l2gx.graphs import TGraph


def metis_clustering(graph: TGraph, num_clusters: int):
    """
    METIS multi-level graph partitioning algorithm
    
    This function uses the METIS library to partition a graph into balanced
    clusters while minimizing the edge cut. METIS is particularly effective
    for creating high-quality partitions with minimal inter-cluster connections.
    
    Args:
        graph: Input graph (TGraph object)
        num_clusters: Number of clusters to create
        
    Returns:
        torch.Tensor: Cluster assignment tensor
        
    Note:
        This function requires a TGraph object because it needs access to
        the adjacency list format (adj_index) that METIS expects.
        
    Example:
        ```python
        from l2gx.patch.clustering.metis import metis_clustering
        from l2gx.graphs import TGraph
        
        # Create TGraph from edge list
        graph = TGraph(edge_index, num_nodes=1000)
        
        # Partition into 8 balanced clusters
        clusters = metis_clustering(graph, num_clusters=8)
        ```
        
    Raises:
        ImportError: If pymetis is not installed
        ValueError: If graph format is incompatible with METIS
    """
    try:
        # Run METIS partitioning
        # pymetis.part_graph returns (edgecuts, membership)
        _, memberships = pymetis.part_graph(
            num_clusters,
            adjncy=graph.edge_index[1],  # Adjacency list
            xadj=graph.adj_index,        # Index into adjacency list
            eweights=graph.edge_attr,    # Edge weights (optional)
        )
        
        # Convert to tensor format
        cluster_tensor = torch.as_tensor(
            memberships, 
            dtype=torch.long, 
            device=graph.device
        )
        
        return cluster_tensor
        
    except ImportError as e:
        raise ImportError(
            "METIS clustering requires the 'pymetis' package. "
            "Install it with: pip install pymetis"
        ) from e


def metis_clustering_with_stats(graph: TGraph, num_clusters: int, verbose=True):
    """
    METIS clustering with additional statistics reporting
    
    Args:
        graph: Input graph (TGraph object)
        num_clusters: Number of clusters to create
        verbose: Whether to print statistics (default: True)
        
    Returns:
        dict: Dictionary containing:
            - 'clusters': Cluster assignment tensor
            - 'edge_cuts': Number of edges cut by the partitioning
            - 'cluster_sizes': Sizes of each cluster
            - 'imbalance': Load imbalance ratio
    """
    try:
        # Run METIS partitioning
        edge_cuts, memberships = pymetis.part_graph(
            num_clusters,
            adjncy=graph.edge_index[1],
            xadj=graph.adj_index,
            eweights=graph.edge_attr,
        )
        
        # Convert to tensor
        cluster_tensor = torch.as_tensor(
            memberships, 
            dtype=torch.long, 
            device=graph.device
        )
        
        # Calculate statistics
        cluster_sizes = torch.bincount(cluster_tensor, minlength=num_clusters)
        avg_size = graph.num_nodes / num_clusters
        max_size = cluster_sizes.max().item()
        imbalance = max_size / avg_size
        
        if verbose:
            print("METIS clustering results:")
            print(f"  Clusters: {num_clusters}")
            print(f"  Edge cuts: {edge_cuts}")
            print(f"  Cluster sizes: {cluster_sizes.tolist()}")
            print(f"  Average size: {avg_size:.1f}")
            print(f"  Imbalance ratio: {imbalance:.3f}")
        
        return {
            'clusters': cluster_tensor,
            'edge_cuts': edge_cuts,
            'cluster_sizes': cluster_sizes,
            'imbalance': imbalance
        }
        
    except ImportError as e:
        raise ImportError(
            "METIS clustering requires the 'pymetis' package. "
            "Install it with: pip install pymetis"
        ) from e


def metis_clustering_weighted(
    graph: TGraph, 
    num_clusters: int, 
    node_weights=None,
    edge_weights=None,
    verbose=True
):
    """
    METIS clustering with custom node and edge weights
    
    Args:
        graph: Input graph (TGraph object)
        num_clusters: Number of clusters to create
        node_weights: Node weights for balanced partitioning (optional)
        edge_weights: Edge weights (optional, uses graph.edge_attr if None)
        verbose: Whether to print statistics
        
    Returns:
        torch.Tensor: Cluster assignment tensor
    """
    try:
        # Use provided edge weights or fall back to graph edge attributes
        eweights = edge_weights
        if eweights is None:
            eweights = graph.edge_attr
            
        # Run METIS with weights
        edge_cuts, memberships = pymetis.part_graph(
            num_clusters,
            adjncy=graph.edge_index[1],
            xadj=graph.adj_index,
            vweights=node_weights,  # Node weights
            eweights=eweights,      # Edge weights
        )
        
        cluster_tensor = torch.as_tensor(
            memberships, 
            dtype=torch.long, 
            device=graph.device
        )
        
        if verbose:
            cluster_sizes = torch.bincount(cluster_tensor, minlength=num_clusters)
            print(f"Weighted METIS clustering: {num_clusters} clusters, "
                  f"{edge_cuts} edge cuts, sizes: {cluster_sizes.tolist()}")
        
        return cluster_tensor
        
    except ImportError as e:
        raise ImportError(
            "METIS clustering requires the 'pymetis' package. "
            "Install it with: pip install pymetis"
        ) from e