"""
Clustering Utilities

This module provides utility classes and functions that are shared across
different clustering algorithms, including partition representations,
clustering evaluation metrics, and format conversion utilities.
"""

import torch
import numpy as np
from typing import Sequence
from torch_geometric.data import Data

from l2gx.graphs import TGraph


class Partition(Sequence):
    """
    Efficient representation of a graph partition

    This class provides a convenient interface for working with cluster assignments,
    allowing easy access to nodes in each cluster and various partition statistics.

    Attributes:
        num_parts (int): Number of clusters/partitions
        nodes (torch.Tensor): Node indices sorted by cluster assignment
        part_index (torch.Tensor): Index array for accessing clusters
    """

    def __init__(self, partition_tensor):
        """
        Initialize partition from cluster assignment tensor

        Args:
            partition_tensor: Tensor of cluster assignments for each node
        """
        partition_tensor = torch.as_tensor(partition_tensor)

        # Count nodes in each cluster
        counts = torch.bincount(partition_tensor)
        self.num_parts = len(counts)

        # Sort nodes by cluster assignment for efficient access
        self.nodes = torch.argsort(partition_tensor)

        # Create index array for O(1) cluster access
        self.part_index = torch.zeros(self.num_parts + 1, dtype=torch.long)
        self.part_index[1:] = torch.cumsum(counts, dim=0)

    def __getitem__(self, item):
        """Get nodes in cluster 'item'"""
        return self.nodes[self.part_index[item] : self.part_index[item + 1]]

    def __len__(self):
        """Number of clusters"""
        return self.num_parts

    def cluster_sizes(self):
        """Get sizes of all clusters"""
        return self.part_index[1:] - self.part_index[:-1]

    def largest_cluster_size(self):
        """Size of the largest cluster"""
        return self.cluster_sizes().max().item()

    def smallest_cluster_size(self):
        """Size of the smallest cluster"""
        return self.cluster_sizes().min().item()

    def imbalance_ratio(self):
        """Ratio of largest to average cluster size"""
        sizes = self.cluster_sizes()
        return sizes.max().float() / sizes.float().mean()

    def to_dict(self):
        """Convert to dictionary mapping cluster_id -> node_list"""
        return {i: self[i].tolist() for i in range(len(self))}


def evaluate_clustering(clusters, graph=None, ground_truth=None):
    """
    Evaluate clustering quality with various metrics

    Args:
        clusters: Cluster assignment tensor
        graph: Optional graph for computing edge cut metrics
        ground_truth: Optional ground truth clustering for comparison metrics

    Returns:
        Dictionary of evaluation metrics
    """
    partition = Partition(clusters)
    metrics = {
        "num_clusters": partition.num_parts,
        "cluster_sizes": partition.cluster_sizes().tolist(),
        "largest_cluster": partition.largest_cluster_size(),
        "smallest_cluster": partition.smallest_cluster_size(),
        "imbalance_ratio": partition.imbalance_ratio(),
        "size_std": partition.cluster_sizes().float().std().item(),
    }

    # Graph-based metrics
    if graph is not None:
        edge_cuts = compute_edge_cuts(clusters, graph)
        metrics.update(
            {
                "total_edge_cuts": edge_cuts.sum().item(),
                "edge_cuts_per_cluster": edge_cuts.tolist(),
                "cut_ratio": edge_cuts.sum().item() / graph.num_edges
                if hasattr(graph, "num_edges")
                else None,
            }
        )

    # Comparison metrics
    if ground_truth is not None:
        comparison_metrics = compare_clusterings(clusters, ground_truth)
        metrics.update(comparison_metrics)

    return metrics


def compute_edge_cuts(clusters, graph):
    """
    Compute number of edges cut by the clustering

    Args:
        clusters: Cluster assignment tensor
        graph: Graph object (TGraph or similar with edge_index)

    Returns:
        torch.Tensor: Number of cut edges for each cluster
    """
    if hasattr(graph, "edge_index"):
        edge_index = graph.edge_index
    else:
        raise ValueError("Graph must have edge_index attribute")

    # Find edges that cross cluster boundaries
    src_clusters = clusters[edge_index[0]]
    dst_clusters = clusters[edge_index[1]]
    cut_edges = src_clusters != dst_clusters

    # Count cuts per cluster (assign to source cluster)
    cuts_per_cluster = torch.zeros(len(torch.unique(clusters)), dtype=torch.long)
    if cut_edges.any():
        cut_src_clusters = src_clusters[cut_edges]
        cuts_per_cluster = torch.bincount(
            cut_src_clusters, minlength=len(cuts_per_cluster)
        )

    return cuts_per_cluster


def compare_clusterings(clusters1, clusters2):
    """
    Compare two clusterings using standard metrics

    Args:
        clusters1: First clustering
        clusters2: Second clustering (e.g., ground truth)

    Returns:
        Dictionary with comparison metrics
    """
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        clusters1_np = (
            clusters1.cpu().numpy()
            if torch.is_tensor(clusters1)
            else np.array(clusters1)
        )
        clusters2_np = (
            clusters2.cpu().numpy()
            if torch.is_tensor(clusters2)
            else np.array(clusters2)
        )

        return {
            "adjusted_rand_score": adjusted_rand_score(clusters1_np, clusters2_np),
            "normalized_mutual_info": normalized_mutual_info_score(
                clusters1_np, clusters2_np
            ),
        }
    except ImportError:
        # Fallback if sklearn not available
        return {
            "adjusted_rand_score": None,
            "normalized_mutual_info": None,
            "note": "sklearn required for comparison metrics",
        }


def clustering_to_patches(clusters, graph, coordinates=None):
    """
    Convert clustering result to patch objects

    Args:
        clusters: Cluster assignment tensor
        graph: Graph object
        coordinates: Optional node coordinates/embeddings

    Returns:
        list of Patch objects
    """
    from l2gx.patch import Patch

    partition = Partition(clusters)
    patches = []

    for i in range(len(partition)):
        cluster_nodes = partition[i]

        if coordinates is not None:
            # Use provided coordinates
            if torch.is_tensor(coordinates):
                cluster_coords = coordinates[cluster_nodes].cpu().numpy()
            else:
                cluster_coords = np.array(coordinates)[cluster_nodes]
        else:
            # Create dummy coordinates if none provided
            cluster_coords = np.random.randn(len(cluster_nodes), 2)

        # Create patch
        patch = Patch(cluster_nodes.cpu().numpy(), cluster_coords)
        patches.append(patch)

    return patches


def convert_graph_format(graph, target_format):
    """
    Convert between different graph formats for clustering algorithms

    Args:
        graph: Input graph in various formats
        target_format: Target format ('tgraph', 'pyg', 'networkx', 'raphtory')

    Returns:
        Graph in target format
    """
    if target_format == "tgraph":
        if hasattr(graph, "edge_index"):
            # Already PyG-like, convert to TGraph
            return TGraph(graph.edge_index, num_nodes=graph.num_nodes)
        else:
            raise NotImplementedError(
                f"Conversion from {type(graph)} to TGraph not implemented"
            )

    elif target_format == "pyg":
        if hasattr(graph, "edge_index"):
            # Already PyG-like
            return Data(edge_index=graph.edge_index, num_nodes=graph.num_nodes)
        else:
            raise NotImplementedError(
                f"Conversion from {type(graph)} to PyG not implemented"
            )

    elif target_format == "networkx":
        import networkx as nx

        if hasattr(graph, "edge_index"):
            G = nx.Graph()
            G.add_nodes_from(range(graph.num_nodes))
            edges = graph.edge_index.t().cpu().numpy()
            G.add_edges_from(edges)
            return G
        else:
            raise NotImplementedError(
                f"Conversion from {type(graph)} to NetworkX not implemented"
            )

    else:
        raise ValueError(f"Unknown target format: {target_format}")


def validate_clustering_result(clusters, num_nodes, num_clusters=None):
    """
    Validate clustering result for consistency

    Args:
        clusters: Cluster assignment tensor/array
        num_nodes: Expected number of nodes
        num_clusters: Expected number of clusters (optional)

    Returns:
        Dictionary with validation results
    """
    clusters = torch.as_tensor(clusters)

    issues = []

    # Check length
    if len(clusters) != num_nodes:
        issues.append(f"Length mismatch: got {len(clusters)}, expected {num_nodes}")

    # Check for negative values
    if (clusters < 0).any():
        num_negative = (clusters < 0).sum().item()
        issues.append(f"{num_negative} nodes have negative cluster assignments")

    # Check cluster ID continuity
    unique_clusters = torch.unique(clusters[clusters >= 0])
    if len(unique_clusters) > 0:
        max_cluster = unique_clusters.max().item()
        expected_range = torch.arange(len(unique_clusters))
        if not torch.equal(unique_clusters, expected_range):
            issues.append("Cluster IDs are not contiguous starting from 0")

    # Check expected number of clusters
    actual_clusters = len(unique_clusters)
    if num_clusters is not None and actual_clusters != num_clusters:
        issues.append(f"Expected {num_clusters} clusters, got {actual_clusters}")

    # Compute basic statistics
    sizes = (
        torch.bincount(clusters[clusters >= 0])
        if len(unique_clusters) > 0
        else torch.tensor([])
    )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "num_clusters": actual_clusters,
        "cluster_sizes": sizes.tolist() if len(sizes) > 0 else [],
        "unassigned_nodes": (clusters < 0).sum().item(),
    }
