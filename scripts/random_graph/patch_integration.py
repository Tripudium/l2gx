#!/usr/bin/env python3
"""
Integration utilities for using random graph models with patch-based algorithms.

This module provides utilities to:
1. Convert community structure to patch format
2. Generate embeddings for communities
3. Create test cases compatible with existing patch alignment pipeline
4. Evaluate patch detection performance on known ground truth
"""

import numpy as np
import networkx as nx
from typing import Optional, set
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from l2gx.patch import Patch
from l2gx.graphs.tgraph import TGraph
import torch

from run.random_graph.agm import CommunityAffiliationGraphModel
from run.random_graph.osbm import OverlappingStochasticBlockModel
from run.random_graph.roc import RandomOverlappingCommunities


class CommunityToPatchConverter:
    """
    Convert community structure from random graph models to patch format
    for testing patch-based algorithms.
    """

    def __init__(self, min_patch_size: int = 10, overlap_threshold: float = 0.5):
        """
        Initialize converter.

        Args:
            min_patch_size: Minimum size for a community to become a patch
            overlap_threshold: Minimum overlap ratio to create patch connections
        """
        self.min_patch_size = min_patch_size
        self.overlap_threshold = overlap_threshold

    def communities_to_patches(
        self,
        graph: nx.Graph,
        community_nodes: dict[int, set[int]],
        node_coordinates: np.ndarray,
    ) -> list[Patch]:
        """
        Convert communities to patches with embeddings.

        Args:
            graph: NetworkX graph
            community_nodes: Dictionary mapping community ID to set of nodes
            node_coordinates: Node coordinates/embeddings (n_nodes x dim)

        Returns:
            list of Patch objects
        """
        patches = []

        for community_id, nodes in community_nodes.items():
            nodes_list = list(nodes)

            # Skip communities that are too small
            if len(nodes_list) < self.min_patch_size:
                continue

            # Extract coordinates for nodes in this community
            patch_coordinates = node_coordinates[nodes_list]

            # Create node index mapping (patch-local index -> global node ID)
            node_index = {i: node_id for i, node_id in enumerate(nodes_list)}

            # Create patch
            patch = Patch(
                nodes=nodes_list, coordinates=patch_coordinates, index=node_index
            )

            patches.append(patch)

        return patches

    def create_patch_graph(self, patches: list[Patch], min_overlap: int = 5) -> TGraph:
        """
        Create patch graph with overlap information.

        Args:
            patches: list of patch objects
            min_overlap: Minimum overlap size to create edge between patches

        Returns:
            TGraph object with patches and overlap information
        """
        n_patches = len(patches)

        # Create patch connectivity based on overlaps
        edges = []
        overlap_nodes = {}

        for i in range(n_patches):
            for j in range(i + 1, n_patches):
                # Find overlapping nodes
                nodes_i = set(patches[i].nodes)
                nodes_j = set(patches[j].nodes)
                overlap = list(nodes_i & nodes_j)

                if len(overlap) >= min_overlap:
                    edges.append((i, j))
                    overlap_nodes[(i, j)] = overlap
                    overlap_nodes[(j, i)] = overlap

        # Create edge_index tensor
        if edges:
            edge_array = np.array(edges).T
            # Make undirected by adding reverse edges
            edge_index = np.concatenate([edge_array, edge_array[[1, 0]]], axis=1)
        else:
            edge_index = np.empty((2, 0), dtype=int)

        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

        # Create TGraph
        patch_graph = TGraph(edge_index=edge_index_tensor, num_nodes=n_patches)

        # Add patches and overlap information
        patch_graph.patches = patches
        patch_graph.overlap_nodes = overlap_nodes

        return patch_graph


def generate_synthetic_embeddings(
    graph: nx.Graph,
    community_nodes: dict[int, set[int]],
    embedding_dim: int = 64,
    community_separation: float = 2.0,
    noise_scale: float = 0.1,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic node embeddings that respect community structure.

    This creates embeddings where nodes in the same community are closer
    in embedding space, making it suitable for testing patch-based algorithms.

    Args:
        graph: NetworkX graph
        community_nodes: Dictionary mapping community ID to nodes
        embedding_dim: Dimension of node embeddings
        community_separation: Distance between community centers
        noise_scale: Amount of random noise to add
        random_state: Random seed

    Returns:
        Node embeddings (n_nodes x embedding_dim)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_nodes = graph.number_of_nodes()
    n_communities = len(community_nodes)

    # Initialize embeddings
    embeddings = np.zeros((n_nodes, embedding_dim))

    # Create community centers in embedding space
    # Place them roughly evenly spaced
    if embedding_dim >= 2:
        # Use first 2 dimensions for community placement
        angles = np.linspace(0, 2 * np.pi, n_communities, endpoint=False)
        community_centers = community_separation * np.column_stack(
            [np.cos(angles), np.sin(angles)]
        )

        # Pad with zeros for higher dimensions
        if embedding_dim > 2:
            padding = np.zeros((n_communities, embedding_dim - 2))
            community_centers = np.column_stack([community_centers, padding])
    else:
        # 1D case: space communities along line
        community_centers = community_separation * np.linspace(
            -1, 1, n_communities
        ).reshape(-1, 1)

    # Add random variation to higher dimensions
    if embedding_dim > 2:
        community_centers[:, 2:] += (
            np.random.randn(n_communities, embedding_dim - 2) * 0.5
        )

    # Assign embeddings based on community membership
    node_assigned = np.zeros(n_nodes, dtype=bool)

    for community_id, nodes in community_nodes.items():
        center = community_centers[community_id]

        for node in nodes:
            if not node_assigned[node]:
                # First community assignment - place near center
                embeddings[node] = center + np.random.randn(embedding_dim) * noise_scale
                node_assigned[node] = True
            else:
                # Node already assigned - blend with existing embedding
                # (for overlapping communities)
                current = embeddings[node]
                target = center + np.random.randn(embedding_dim) * noise_scale
                embeddings[node] = 0.5 * (current + target)

    # Handle any unassigned nodes (shouldn't happen with valid communities)
    unassigned = ~node_assigned
    if np.any(unassigned):
        embeddings[unassigned] = (
            np.random.randn(np.sum(unassigned), embedding_dim) * noise_scale
        )

    return embeddings


class CommunityTestCase:
    """
    Complete test case combining graph structure, communities, and embeddings.
    """

    def __init__(
        self,
        model_type: str,
        model_params: dict,
        embedding_params: dict = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize test case.

        Args:
            model_type: "agm", "osbm", or "roc"
            model_params: Parameters for the graph model
            embedding_params: Parameters for embedding generation
            random_state: Random seed
        """
        self.model_type = model_type.lower()
        self.model_params = model_params
        self.embedding_params = embedding_params or {}
        self.random_state = random_state

        # set default embedding parameters
        self.embedding_params.setdefault("embedding_dim", 64)
        self.embedding_params.setdefault("community_separation", 2.0)
        self.embedding_params.setdefault("noise_scale", 0.1)

        # Will be populated during generation
        self.model = None
        self.graph = None
        self.embeddings = None
        self.patches = None
        self.patch_graph = None

    def generate(self) -> tuple[nx.Graph, np.ndarray, list[Patch], TGraph]:
        """
        Generate complete test case.

        Returns:
            tuple of (graph, embeddings, patches, patch_graph)
        """
        # Generate graph model
        if self.model_type == "agm":
            self.model = CommunityAffiliationGraphModel(
                random_state=self.random_state, **self.model_params
            )
        elif self.model_type == "osbm":
            self.model = OverlappingStochasticBlockModel(
                random_state=self.random_state, **self.model_params
            )
        elif self.model_type == "roc":
            self.model = RandomOverlappingCommunities(
                random_state=self.random_state, **self.model_params
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Generate graph
        self.graph = self.model.generate_graph()

        # Generate embeddings
        community_nodes = self.model.get_community_nodes()
        self.embeddings = generate_synthetic_embeddings(
            self.graph,
            community_nodes,
            random_state=self.random_state,
            **self.embedding_params,
        )

        # Convert to patches
        converter = CommunityToPatchConverter()
        self.patches = converter.communities_to_patches(
            self.graph, community_nodes, self.embeddings
        )

        # Create patch graph
        self.patch_graph = converter.create_patch_graph(self.patches)

        return self.graph, self.embeddings, self.patches, self.patch_graph

    def get_ground_truth_communities(self) -> dict[int, set[int]]:
        """Get ground truth community assignments."""
        if self.model is None:
            raise ValueError("Must generate test case first")
        return self.model.get_community_nodes()

    def evaluate_patch_recovery(self, detected_patches: list[Patch]) -> dict:
        """
        Evaluate how well detected patches match ground truth communities.

        Args:
            detected_patches: Patches detected by algorithm

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Must generate test case first")

        true_communities = self.get_ground_truth_communities()

        # Convert detected patches to community format
        detected_communities = {}
        for i, patch in enumerate(detected_patches):
            detected_communities[i] = set(patch.nodes)

        # Compute evaluation metrics
        metrics = {}

        # Jaccard similarity for best matches
        similarities = []
        for true_id, true_nodes in true_communities.items():
            best_jaccard = 0.0
            for det_id, det_nodes in detected_communities.items():
                intersection = len(true_nodes & det_nodes)
                union = len(true_nodes | det_nodes)
                jaccard = intersection / union if union > 0 else 0.0
                best_jaccard = max(best_jaccard, jaccard)
            similarities.append(best_jaccard)

        metrics["avg_jaccard"] = np.mean(similarities)
        metrics["min_jaccard"] = np.min(similarities)
        metrics["communities_recovered"] = sum(1 for s in similarities if s > 0.5)
        metrics["recovery_rate"] = metrics["communities_recovered"] / len(
            true_communities
        )

        return metrics


def create_standard_test_suite() -> list[CommunityTestCase]:
    """Create a standard suite of test cases for patch-based algorithm evaluation."""

    test_suite = [
        # AGM test cases
        CommunityTestCase(
            "agm",
            {
                "n_nodes": 150,
                "n_communities": 6,
                "community_size_dist": "uniform",
                "community_size_params": {"min_size": 20, "max_size": 35},
                "intra_community_prob": 0.3,
                "background_prob": 0.01,
            },
            {"embedding_dim": 32, "community_separation": 3.0},
        ),
        # OSBM test cases
        CommunityTestCase(
            "osbm",
            {
                "n_nodes": 120,
                "n_communities": 5,
                "community_assignments": "balanced",
                "overlap_probability": 0.2,
                "intra_community_prob": 0.35,
                "inter_community_prob": 0.02,
            },
            {"embedding_dim": 64, "community_separation": 2.5},
        ),
        # ROC test cases
        CommunityTestCase(
            "roc",
            {
                "n_nodes": 200,
                "n_communities": 8,
                "community_generation": "preferential",
                "average_community_size": 25,
                "overlap_factor": 0.3,
                "membership_strength_dist": "beta",
                "base_edge_prob": 0.15,
            },
            {"embedding_dim": 128, "community_separation": 2.0},
        ),
    ]

    return test_suite


def demo_patch_integration():
    """Demonstrate integration with patch-based algorithms."""
    print("PATCH INTEGRATION DEMO")
    print("=" * 40)

    # Create a test case
    test_case = CommunityTestCase(
        "agm",
        {
            "n_nodes": 80,
            "n_communities": 4,
            "community_size_dist": "uniform",
            "community_size_params": {"min_size": 15, "max_size": 25},
            "intra_community_prob": 0.4,
            "background_prob": 0.01,
        },
        {"embedding_dim": 32},
        random_state=42,
    )

    # Generate test case
    graph, embeddings, patches, patch_graph = test_case.generate()

    print("Generated test case:")
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Patches: {len(patches)} patches")
    print(
        f"  Patch graph: {patch_graph.num_nodes} nodes, {patch_graph.edge_index.shape[1] // 2} edges"
    )

    # Show patch statistics
    patch_sizes = [len(p.nodes) for p in patches]
    print(
        f"  Patch sizes: min={min(patch_sizes)}, max={max(patch_sizes)}, avg={np.mean(patch_sizes):.1f}"
    )

    # This patch_graph can now be used with existing alignment algorithms
    print("\nThis patch_graph is compatible with:")
    print("  - get_aligner('l2g').align_patches(patch_graph)")
    print("  - get_aligner('geo').align_patches(patch_graph)")
    print("  - Evaluation against ground truth communities")


if __name__ == "__main__":
    demo_patch_integration()
