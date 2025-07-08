#!/usr/bin/env python3
"""
Test Graph Generation Utilities

This module provides utilities for generating test graphs for benchmarking
and testing purposes.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2gx.graphs import TGraph


def generate_test_graph(num_nodes: int, avg_degree: int = 10, in_cluster_prob: float = 0.7, seed: int = 42) -> TGraph:
    """Generate a random test graph with specified properties"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random edges
    num_edges = (num_nodes * avg_degree) // 2
    
    # Create edges with some structure (not completely random)
    edges = []
    
    # Add some structured clusters
    cluster_size = num_nodes // 10
    for cluster_start in range(0, num_nodes, cluster_size):
        cluster_end = min(cluster_start + cluster_size, num_nodes)
        
        # Dense connections within cluster
        for i in range(cluster_start, cluster_end):
            for j in range(i + 1, min(i + avg_degree // 2, cluster_end)):
                if np.random.random() < in_cluster_prob:
                    edges.append([i, j])
    
    # Add some random inter-cluster edges
    for _ in range(num_edges // 3):
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j:
            edges.append([i, j])
    
    # Remove duplicates and convert to tensor
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Sort edges by source node (important for adjacency index construction)
    index = torch.argsort(edge_index[0] * num_nodes + edge_index[1])
    edge_index = edge_index[:, index]
    
    return TGraph(edge_index, num_nodes=num_nodes)

def generate_hidden_partition_model(num_nodes: int, num_clusters: int, in_cluster_prob: float = 0.7, out_cluster_prob: float = 0.01, seed: int = 42) -> TGraph:
    """Generate a hidden partition model graph with specified properties"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate random cluster assignments
    cluster_assignments = np.random.randint(0, num_clusters, size=num_nodes)
    
    # Create edges within clusters
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cluster_assignments[i] == cluster_assignments[j]:
                if np.random.random() < in_cluster_prob:
                    edges.append([i, j])
            else:
                if np.random.random() < out_cluster_prob:
                    edges.append([i, j])
    
    # Remove duplicates and convert to tensor
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # Sort edges by source node (important for adjacency index construction)
    index = torch.argsort(edge_index[0] * num_nodes + edge_index[1])
    edge_index = edge_index[:, index]
    
    return TGraph(edge_index, num_nodes=num_nodes), cluster_assignments
    
    