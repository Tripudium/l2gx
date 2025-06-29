"""
Fennel Clustering Utilities

This module provides easy-to-use wrapper functions for Fennel clustering
that work with different graph representations without requiring Raphtory.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from .clustering import _fennel_clustering
from ..graphs import TGraph


def fennel_clustering_from_tgraph(
    graph: TGraph, 
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1
) -> torch.Tensor:
    """
    Run Fennel clustering on a TGraph object.
    
    Args:
        graph: TGraph object
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
    
    Returns:
        Cluster assignment tensor
    """
    # Extract required data from TGraph
    edge_index_np = graph.edge_index.cpu().numpy()
    adj_index_np = graph.adj_index.cpu().numpy()
    num_nodes = graph.num_nodes
    
    # Run Fennel clustering
    clusters = _fennel_clustering(
        edge_index=edge_index_np,
        adj_index=adj_index_np,
        num_nodes=num_nodes,
        num_clusters=num_clusters,
        load_limit=load_limit,
        alpha=alpha,
        gamma=gamma,
        num_iters=num_iters
    )
    
    return torch.tensor(clusters, dtype=torch.long)


def fennel_clustering_from_pyg(
    data: Data,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1
) -> torch.Tensor:
    """
    Run Fennel clustering on a PyTorch Geometric Data object.
    
    Args:
        data: PyTorch Geometric Data object
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
    
    Returns:
        Cluster assignment tensor
    """
    # Convert to TGraph first (this handles the adjacency index computation)
    tgraph = TGraph(data.edge_index, edge_attr=data.edge_attr, x=data.x, num_nodes=data.num_nodes)
    
    # Use the TGraph version
    return fennel_clustering_from_tgraph(
        tgraph, num_clusters, load_limit, alpha, gamma, num_iters
    )


def fennel_clustering_from_edge_list(
    edge_index: np.ndarray,
    num_nodes: int,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1
) -> np.ndarray:
    """
    Run Fennel clustering on raw edge list data.
    
    Args:
        edge_index: Edge list as numpy array (shape: [2, num_edges])
        num_nodes: Number of nodes in the graph
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
    
    Returns:
        Cluster assignment array
    """
    # Create TGraph to compute adjacency index
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)
    tgraph = TGraph(edge_index_torch, num_nodes=num_nodes)
    
    # Extract data and run clustering
    adj_index_np = tgraph.adj_index.cpu().numpy()
    
    clusters = _fennel_clustering(
        edge_index=edge_index,
        adj_index=adj_index_np,
        num_nodes=num_nodes,
        num_clusters=num_clusters,
        load_limit=load_limit,
        alpha=alpha,
        gamma=gamma,
        num_iters=num_iters
    )
    
    return clusters


def fennel_clustering_from_networkx(
    nx_graph,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1
) -> dict:
    """
    Run Fennel clustering on a NetworkX graph.
    
    Args:
        nx_graph: NetworkX graph object
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
    
    Returns:
        Dictionary mapping node IDs to cluster IDs
    """
    import networkx as nx
    
    # Convert NetworkX to edge list
    num_nodes = nx_graph.number_of_nodes()
    
    # Create node mapping if nodes are not 0-indexed integers
    nodes = list(nx_graph.nodes())
    if nodes != list(range(num_nodes)):
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        idx_to_node = {idx: node for idx, node in enumerate(nodes)}
    else:
        node_to_idx = {i: i for i in range(num_nodes)}
        idx_to_node = {i: i for i in range(num_nodes)}
    
    # Convert edges
    edges = []
    for src, dst in nx_graph.edges():
        edges.append([node_to_idx[src], node_to_idx[dst]])
    
    if not edges:
        # Handle empty graph
        return {node: 0 for node in nodes}
    
    edge_index = np.array(edges).T
    
    # For undirected graphs, add reverse edges
    if not nx_graph.is_directed():
        edge_index = np.concatenate([edge_index, edge_index[[1, 0], :]], axis=1)
    
    # Run clustering
    clusters = fennel_clustering_from_edge_list(
        edge_index, num_nodes, num_clusters, load_limit, alpha, gamma, num_iters
    )
    
    # Convert back to original node IDs
    result = {}
    for idx, cluster_id in enumerate(clusters):
        original_node = idx_to_node[idx]
        result[original_node] = int(cluster_id)
    
    return result


# Convenience function that auto-detects input type
def fennel_clustering(
    graph,
    num_clusters: int,
    load_limit: float = 1.1,
    alpha: float = None,
    gamma: float = 1.5,
    num_iters: int = 1
):
    """
    Auto-detecting Fennel clustering function.
    
    Automatically detects the input type and calls the appropriate
    Fennel clustering function.
    
    Args:
        graph: Graph in various formats (TGraph, Data, NetworkX, etc.)
        num_clusters: Target number of clusters
        load_limit: Maximum cluster size factor (default: 1.1)
        alpha: Alpha parameter (computed automatically if None)
        gamma: Gamma parameter (default: 1.5)
        num_iters: Number of iterations (default: 1)
    
    Returns:
        Cluster assignments in appropriate format
    """
    if isinstance(graph, TGraph):
        return fennel_clustering_from_tgraph(graph, num_clusters, load_limit, alpha, gamma, num_iters)
    elif isinstance(graph, Data):
        return fennel_clustering_from_pyg(graph, num_clusters, load_limit, alpha, gamma, num_iters)
    elif hasattr(graph, 'nodes') and hasattr(graph, 'edges'):  # NetworkX-like
        return fennel_clustering_from_networkx(graph, num_clusters, load_limit, alpha, gamma, num_iters)
    else:
        raise TypeError(f"Unsupported graph type: {type(graph)}")