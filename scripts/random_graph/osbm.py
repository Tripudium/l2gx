"""
Overlapping Stochastic Block Model (OSBM)

The OSBM extends the classical Stochastic Block Model to allow nodes to belong to
multiple communities. Edge probabilities depend on the communities that nodes belong to.

Key features:
1. Nodes can belong to multiple overlapping communities
2. Edge probability between nodes depends on their shared community memberships
3. Flexible mixing parameters control intra vs inter-community connectivity
4. Supports various community assignment schemes

This model is excellent for patch-based testing because:
- Overlapping communities create natural patch boundaries
- Edge probabilities create realistic community structure
- Controllable overlap allows testing different patch overlap scenarios

References:
- Latouche, P., Birmelé, E., & Ambroise, C. (2011). Overlapping stochastic block models. Annals of Applied Statistics.
- Ball, B., Karrer, B., & Newman, M. E. J. (2011). Efficient and principled method for detecting communities. Physical Review E.
"""

import numpy as np
import networkx as nx
from typing import Optional, dict, set, Union
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.special import comb


class OverlappingStochasticBlockModel:
    """
    Overlapping Stochastic Block Model for generating graphs with overlapping block structure.

    In this model, nodes belong to one or more communities, and edge probabilities
    depend on the community memberships of the endpoint nodes.
    """

    def __init__(
        self,
        n_nodes: int,
        n_communities: int,
        community_assignments: Union[str, np.ndarray] = "random",
        mixing_matrix: Optional[np.ndarray] = None,
        overlap_probability: float = 0.1,
        community_size_balance: float = 1.0,
        intra_community_prob: float = 0.3,
        inter_community_prob: float = 0.05,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the OSBM model.

        Args:
            n_nodes: Number of nodes in the graph
            n_communities: Number of communities
            community_assignments: How to assign nodes to communities
                                  ("random", "balanced", "power_law", or custom array)
            mixing_matrix: Custom K x K matrix of edge probabilities between communities
                          If None, uses intra/inter community probabilities
            overlap_probability: Probability that a node belongs to multiple communities
            community_size_balance: Controls relative community sizes (1.0 = balanced)
            intra_community_prob: Edge probability within communities
            inter_community_prob: Edge probability between different communities
            random_state: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.community_assignments = community_assignments
        self.overlap_probability = overlap_probability
        self.community_size_balance = community_size_balance
        self.intra_community_prob = intra_community_prob
        self.inter_community_prob = inter_community_prob

        if random_state is not None:
            np.random.seed(random_state)

        # set up mixing matrix
        if mixing_matrix is not None:
            if mixing_matrix.shape != (n_communities, n_communities):
                raise ValueError(
                    "Mixing matrix must be K x K where K is number of communities"
                )
            self.mixing_matrix = mixing_matrix
        else:
            self.mixing_matrix = self._create_default_mixing_matrix()

        # Will be populated during generation
        self.node_communities = {}
        self.community_nodes = defaultdict(set)
        self.graph = None

    def _create_default_mixing_matrix(self) -> np.ndarray:
        """Create default mixing matrix with intra/inter community probabilities."""
        mixing = np.full(
            (self.n_communities, self.n_communities), self.inter_community_prob
        )
        np.fill_diagonal(mixing, self.intra_community_prob)
        return mixing

    def _assign_communities(self) -> None:
        """Assign nodes to communities according to the specified scheme."""

        if isinstance(self.community_assignments, str):
            if self.community_assignments == "random":
                # Each node independently joins each community with some probability
                for node in range(self.n_nodes):
                    communities = set()
                    for c in range(self.n_communities):
                        # Base probability adjusted by balance parameter
                        prob = (1.0 / self.n_communities) * self.community_size_balance
                        if np.random.random() < prob:
                            communities.add(c)

                    # Ensure each node is in at least one community
                    if not communities:
                        communities.add(np.random.randint(self.n_communities))

                    # Add overlap with specified probability
                    if (
                        len(communities) == 1
                        and np.random.random() < self.overlap_probability
                    ):
                        additional_communities = np.random.choice(
                            [
                                c
                                for c in range(self.n_communities)
                                if c not in communities
                            ],
                            size=np.random.randint(1, min(3, self.n_communities)),
                            replace=False,
                        )
                        communities.update(additional_communities)

                    self.node_communities[node] = communities

            elif self.community_assignments == "balanced":
                # Try to create balanced community sizes with controlled overlap
                nodes_per_community = self.n_nodes // self.n_communities

                # First, assign each node to exactly one community
                for node in range(self.n_nodes):
                    primary_community = node % self.n_communities
                    self.node_communities[node] = {primary_community}

                # Then add overlaps
                for node in range(self.n_nodes):
                    if np.random.random() < self.overlap_probability:
                        # Add 1-2 additional communities
                        n_additional = np.random.randint(1, min(3, self.n_communities))
                        current_communities = self.node_communities[node]
                        available = [
                            c
                            for c in range(self.n_communities)
                            if c not in current_communities
                        ]

                        if available:
                            additional = np.random.choice(
                                available,
                                size=min(n_additional, len(available)),
                                replace=False,
                            )
                            self.node_communities[node].update(additional)

            elif self.community_assignments == "power_law":
                # Community sizes follow power law, with overlap
                # First determine community sizes
                alpha = 2.0  # Power law exponent
                sizes = []
                remaining = self.n_nodes

                for c in range(self.n_communities - 1):
                    # Power law distributed size
                    max_size = remaining - (self.n_communities - c - 1)
                    u = np.random.uniform(0, 1)
                    size = max(
                        1, int((max_size ** (1 - alpha)) * u) ** (1 / (1 - alpha))
                    )
                    size = min(size, max_size)
                    sizes.append(size)
                    remaining -= size

                sizes.append(remaining)  # Last community gets remainder

                # Assign nodes to communities
                node_idx = 0
                for c, size in enumerate(sizes):
                    community_nodes = list(
                        range(node_idx, min(node_idx + size, self.n_nodes))
                    )
                    for node in community_nodes:
                        if node not in self.node_communities:
                            self.node_communities[node] = set()
                        self.node_communities[node].add(c)
                    node_idx += size

                # Add overlaps
                for node in range(self.n_nodes):
                    if np.random.random() < self.overlap_probability:
                        current = list(self.node_communities[node])
                        available = [
                            c for c in range(self.n_communities) if c not in current
                        ]
                        if available:
                            additional = np.random.choice(
                                available, size=min(2, len(available)), replace=False
                            )
                            self.node_communities[node].update(additional)

        elif isinstance(self.community_assignments, np.ndarray):
            # Custom community assignment matrix (nodes x communities)
            if self.community_assignments.shape != (self.n_nodes, self.n_communities):
                raise ValueError("Custom assignments must be n_nodes x n_communities")

            for node in range(self.n_nodes):
                communities = set(np.where(self.community_assignments[node, :])[0])
                if not communities:  # Ensure each node is in at least one community
                    communities.add(np.random.randint(self.n_communities))
                self.node_communities[node] = communities

        # Populate reverse mapping
        for node, communities in self.node_communities.items():
            for c in communities:
                self.community_nodes[c].add(node)

    def _compute_edge_probability(self, node_i: int, node_j: int) -> float:
        """
        Compute edge probability between two nodes based on their community memberships.

        For OSBM, there are different ways to combine probabilities from multiple communities:
        1. Maximum: max over shared communities
        2. Average: average over shared communities
        3. Noisy-OR: 1 - prod(1 - p_k) for shared communities k
        4. Sum (capped at 1): sum of probabilities capped at 1
        """
        communities_i = self.node_communities[node_i]
        communities_j = self.node_communities[node_j]

        # Find all pairs of communities (one from each node)
        probs = []
        for c_i in communities_i:
            for c_j in communities_j:
                probs.append(self.mixing_matrix[c_i, c_j])

        if not probs:
            return 0.0

        # Use noisy-OR combination (common in overlapping models)
        # P(edge) = 1 - prod(1 - p_ij) over all community pairs
        prob_no_edge = 1.0
        for p in probs:
            prob_no_edge *= 1.0 - p

        return 1.0 - prob_no_edge

    def generate_graph(self) -> nx.Graph:
        """
        Generate a graph using the OSBM model.

        Returns:
            NetworkX graph with overlapping block structure
        """
        # Step 1: Assign nodes to communities
        self._assign_communities()

        # Step 2: Generate edges based on community memberships
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                edge_prob = self._compute_edge_probability(i, j)

                if np.random.random() < edge_prob:
                    G.add_edge(i, j)

        # Add community information as node attributes
        for node in G.nodes():
            G.nodes[node]["communities"] = list(self.node_communities[node])

        self.graph = G
        return G

    def get_community_memberships(self) -> dict[int, set[int]]:
        """Get the community membership for each node."""
        return dict(self.node_communities)

    def get_community_nodes(self) -> dict[int, set[int]]:
        """Get the nodes in each community."""
        return dict(self.community_nodes)

    def compute_statistics(self) -> dict:
        """Compute various statistics about the generated graph."""
        if self.graph is None:
            raise ValueError("Must generate graph first")

        stats = {}

        # Basic graph statistics
        stats["n_nodes"] = self.graph.number_of_nodes()
        stats["n_edges"] = self.graph.number_of_edges()
        stats["density"] = nx.density(self.graph)

        # Community statistics
        stats["n_communities"] = self.n_communities
        community_sizes = [
            len(self.community_nodes[c]) for c in range(self.n_communities)
        ]
        stats["community_size_mean"] = np.mean(community_sizes)
        stats["community_size_std"] = np.std(community_sizes)
        stats["community_size_min"] = np.min(community_sizes)
        stats["community_size_max"] = np.max(community_sizes)

        # Overlap statistics
        node_memberships = [
            len(self.node_communities[node]) for node in range(self.n_nodes)
        ]
        stats["avg_communities_per_node"] = np.mean(node_memberships)
        stats["max_communities_per_node"] = np.max(node_memberships)
        stats["nodes_in_multiple_communities"] = sum(
            1 for m in node_memberships if m > 1
        )
        stats["overlap_ratio"] = stats["nodes_in_multiple_communities"] / self.n_nodes

        # Clustering and modularity
        stats["clustering_coefficient"] = nx.average_clustering(self.graph)

        # Community-specific edge statistics
        intra_edges = 0
        inter_edges = 0

        for i, j in self.graph.edges():
            communities_i = self.node_communities[i]
            communities_j = self.node_communities[j]

            if communities_i & communities_j:  # Shared community
                intra_edges += 1
            else:
                inter_edges += 1

        stats["intra_community_edges"] = intra_edges
        stats["inter_community_edges"] = inter_edges
        if intra_edges + inter_edges > 0:
            stats["intra_edge_fraction"] = intra_edges / (intra_edges + inter_edges)
        else:
            stats["intra_edge_fraction"] = 0.0

        return stats

    def visualize(
        self,
        layout: str = "spring",
        node_size: int = 100,
        figsize: tuple[int, int] = (15, 5),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the generated graph with community structure.

        Args:
            layout: Layout algorithm ("spring", "circular", "kamada_kawai")
            node_size: Size of nodes in visualization
            figsize: Figure size
            save_path: Optional path to save the visualization
        """
        if self.graph is None:
            raise ValueError("Must generate graph first")

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Plot 1: Color by number of communities
        node_colors = [len(self.node_communities[node]) for node in self.graph.nodes()]
        im1 = nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            with_labels=False,
            cmap="viridis",
            ax=axes[0],
        )
        axes[0].set_title("Nodes by # communities")

        # Plot 2: Show specific community
        if self.n_communities > 0:
            target_community = 0  # Show first community
            node_colors = [
                "red"
                if target_community in self.node_communities[node]
                else "lightblue"
                for node in self.graph.nodes()
            ]
            nx.draw(
                self.graph,
                pos,
                node_color=node_colors,
                node_size=node_size,
                with_labels=False,
                ax=axes[1],
            )
            axes[1].set_title(f"Community {target_community} highlighted")

        # Plot 3: Show overlapping nodes
        node_colors = [
            "orange" if len(self.node_communities[node]) > 1 else "lightblue"
            for node in self.graph.nodes()
        ]
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            with_labels=False,
            ax=axes[2],
        )
        axes[2].set_title("Overlapping nodes (orange)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def demo_osbm():
    """Demonstrate the OSBM model with different parameter settings."""
    print("OVERLAPPING STOCHASTIC BLOCK MODEL DEMO")
    print("=" * 50)

    # Test different configurations
    configs = [
        {
            "name": "Balanced Communities",
            "params": {
                "n_nodes": 120,
                "n_communities": 4,
                "community_assignments": "balanced",
                "overlap_probability": 0.2,
                "intra_community_prob": 0.4,
                "inter_community_prob": 0.05,
            },
        },
        {
            "name": "High Overlap",
            "params": {
                "n_nodes": 100,
                "n_communities": 5,
                "community_assignments": "random",
                "overlap_probability": 0.4,
                "intra_community_prob": 0.3,
                "inter_community_prob": 0.02,
            },
        },
        {
            "name": "Power-law Community Sizes",
            "params": {
                "n_nodes": 150,
                "n_communities": 6,
                "community_assignments": "power_law",
                "overlap_probability": 0.15,
                "intra_community_prob": 0.35,
                "inter_community_prob": 0.03,
            },
        },
    ]

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * len(config["name"]))

        # Generate graph
        osbm = OverlappingStochasticBlockModel(random_state=42, **config["params"])
        G = osbm.generate_graph()

        # Compute statistics
        stats = osbm.compute_statistics()

        print(f"Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}")
        print(f"Density: {stats['density']:.4f}")
        print(f"Communities: {stats['n_communities']}")
        print(
            f"Avg community size: {stats['community_size_mean']:.1f} ± {stats['community_size_std']:.1f}"
        )
        print(f"Avg communities per node: {stats['avg_communities_per_node']:.2f}")
        print(f"Overlap ratio: {stats['overlap_ratio']:.2f}")
        print(f"Intra-community edge fraction: {stats['intra_edge_fraction']:.3f}")
        print(f"Clustering coefficient: {stats['clustering_coefficient']:.3f}")


if __name__ == "__main__":
    demo_osbm()
