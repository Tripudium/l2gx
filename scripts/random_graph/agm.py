"""
Community-Affiliation Graph Model (AGM)

The AGM model generates graphs with overlapping communities by:
1. Creating a bipartite affiliation graph between nodes and communities
2. Connecting nodes that share community memberships with probability p_c
3. Adding random edges with background probability p_b

This model is particularly good for testing patch-based algorithms because:
- Communities naturally form dense, overlapping patches
- Background edges create realistic noise
- Community overlap structure is controllable

References:
- Yang, J., McAuley, J., & Leskovec, J. (2013). Community detection in networks with node attributes. ICDM.
- Fortunato, S., & Hric, D. (2016). Community detection in networks: A user guide. Physics Reports.
"""

import numpy as np
import networkx as nx
from typing import Optional, dict, set
import matplotlib.pyplot as plt
from collections import defaultdict


class CommunityAffiliationGraphModel:
    """
    Community-Affiliation Graph Model for generating graphs with overlapping communities.

    This model first generates a bipartite affiliation between nodes and communities,
    then creates edges based on shared community memberships.
    """

    def __init__(
        self,
        n_nodes: int,
        n_communities: int,
        community_size_dist: str = "poisson",
        community_size_params: dict = None,
        membership_prob: float = 0.1,
        intra_community_prob: float = 0.3,
        background_prob: float = 0.01,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the AGM model.

        Args:
            n_nodes: Number of nodes in the graph
            n_communities: Number of communities
            community_size_dist: Distribution for community sizes ("poisson", "uniform", "power_law")
            community_size_params: Parameters for community size distribution
            membership_prob: Probability that a node belongs to any given community
            intra_community_prob: Probability of edge between nodes in same community
            background_prob: Probability of random background edges
            random_state: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.community_size_dist = community_size_dist
        self.community_size_params = community_size_params or {}
        self.membership_prob = membership_prob
        self.intra_community_prob = intra_community_prob
        self.background_prob = background_prob

        if random_state is not None:
            np.random.seed(random_state)

        # Will be populated during generation
        self.node_communities = defaultdict(set)
        self.community_nodes = defaultdict(set)
        self.graph = None

    def _generate_community_memberships(self) -> None:
        """Generate the bipartite affiliation between nodes and communities."""

        if self.community_size_dist == "uniform":
            # Each community gets approximately the same number of nodes
            min_size = self.community_size_params.get("min_size", 10)
            max_size = self.community_size_params.get("max_size", 50)

            for c in range(self.n_communities):
                size = np.random.randint(min_size, max_size + 1)
                members = np.random.choice(self.n_nodes, size=size, replace=False)
                for node in members:
                    self.node_communities[node].add(c)
                    self.community_nodes[c].add(node)

        elif self.community_size_dist == "poisson":
            # Community sizes follow Poisson distribution
            lambda_param = self.community_size_params.get("lambda", 20)

            for c in range(self.n_communities):
                size = max(1, np.random.poisson(lambda_param))
                size = min(size, self.n_nodes)  # Cap at total nodes
                members = np.random.choice(self.n_nodes, size=size, replace=False)
                for node in members:
                    self.node_communities[node].add(c)
                    self.community_nodes[c].add(node)

        elif self.community_size_dist == "power_law":
            # Community sizes follow power law distribution
            alpha = self.community_size_params.get("alpha", 2.5)
            min_size = self.community_size_params.get("min_size", 5)
            max_size = self.community_size_params.get("max_size", 100)

            for c in range(self.n_communities):
                # Generate power law distributed size
                u = np.random.uniform(0, 1)
                size = int(min_size * (1 - u) ** (-1 / (alpha - 1)))
                size = min(max(size, min_size), min(max_size, self.n_nodes))

                members = np.random.choice(self.n_nodes, size=size, replace=False)
                for node in members:
                    self.node_communities[node].add(c)
                    self.community_nodes[c].add(node)

        elif self.community_size_dist == "bernoulli":
            # Independent Bernoulli membership (classic AGM)
            for node in range(self.n_nodes):
                for c in range(self.n_communities):
                    if np.random.random() < self.membership_prob:
                        self.node_communities[node].add(c)
                        self.community_nodes[c].add(node)
        else:
            raise ValueError(
                f"Unknown community size distribution: {self.community_size_dist}"
            )

    def _generate_edges(self) -> nx.Graph:
        """Generate edges based on community memberships and background noise."""
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))

        # Add edges within communities
        for c in range(self.n_communities):
            community_members = list(self.community_nodes[c])

            # Add edges between all pairs in this community with probability p_c
            for i in range(len(community_members)):
                for j in range(i + 1, len(community_members)):
                    node_i, node_j = community_members[i], community_members[j]

                    if np.random.random() < self.intra_community_prob:
                        G.add_edge(node_i, node_j)

        # Add background edges
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if not G.has_edge(i, j):  # Don't override community edges
                    if np.random.random() < self.background_prob:
                        G.add_edge(i, j)

        return G

    def generate_graph(self) -> nx.Graph:
        """
        Generate a graph using the AGM model.

        Returns:
            NetworkX graph with overlapping community structure
        """
        # Step 1: Generate community memberships
        self._generate_community_memberships()

        # Step 2: Generate edges based on memberships
        self.graph = self._generate_edges()

        # Add community information as node attributes
        for node in self.graph.nodes():
            self.graph.nodes[node]["communities"] = list(self.node_communities[node])

        return self.graph

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
        node_degrees = [
            len(self.node_communities[node]) for node in range(self.n_nodes)
        ]
        stats["avg_communities_per_node"] = np.mean(node_degrees)
        stats["max_communities_per_node"] = np.max(node_degrees)
        stats["nodes_in_multiple_communities"] = sum(1 for d in node_degrees if d > 1)

        # Clustering coefficient
        stats["clustering_coefficient"] = nx.average_clustering(self.graph)

        return stats

    def visualize(
        self,
        layout: str = "spring",
        node_size: int = 100,
        figsize: tuple[int, int] = (12, 8),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize the generated graph with community colors.

        Args:
            layout: Layout algorithm ("spring", "circular", "random")
            node_size: Size of nodes in visualization
            figsize: Figure size
            save_path: Optional path to save the visualization
        """
        if self.graph is None:
            raise ValueError("Must generate graph first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "random":
            pos = nx.random_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Plot 1: Color nodes by number of communities they belong to
        node_colors = [len(self.node_communities[node]) for node in self.graph.nodes()]
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            with_labels=False,
            cmap="viridis",
            ax=ax1,
        )
        ax1.set_title("Nodes colored by # of communities")

        # Plot 2: Highlight largest community
        if self.n_communities > 0:
            largest_community = max(
                range(self.n_communities), key=lambda c: len(self.community_nodes[c])
            )
            node_colors = [
                "red"
                if node in self.community_nodes[largest_community]
                else "lightblue"
                for node in self.graph.nodes()
            ]
            nx.draw(
                self.graph,
                pos,
                node_color=node_colors,
                node_size=node_size,
                with_labels=False,
                ax=ax2,
            )
            ax2.set_title(
                f"Largest community highlighted (size: {len(self.community_nodes[largest_community])})"
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def demo_agm():
    """Demonstrate the AGM model with different parameter settings."""
    print("COMMUNITY-AFFILIATION GRAPH MODEL DEMO")
    print("=" * 50)

    # Test different configurations
    configs = [
        {
            "name": "Small Dense Communities",
            "params": {
                "n_nodes": 100,
                "n_communities": 5,
                "community_size_dist": "uniform",
                "community_size_params": {"min_size": 15, "max_size": 25},
                "intra_community_prob": 0.4,
                "background_prob": 0.01,
            },
        },
        {
            "name": "Many Overlapping Communities",
            "params": {
                "n_nodes": 200,
                "n_communities": 15,
                "community_size_dist": "poisson",
                "community_size_params": {"lambda": 20},
                "intra_community_prob": 0.2,
                "background_prob": 0.005,
            },
        },
        {
            "name": "Power-law Community Sizes",
            "params": {
                "n_nodes": 150,
                "n_communities": 8,
                "community_size_dist": "power_law",
                "community_size_params": {"alpha": 2.5, "min_size": 5, "max_size": 40},
                "intra_community_prob": 0.3,
                "background_prob": 0.01,
            },
        },
    ]

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * len(config["name"]))

        # Generate graph
        agm = CommunityAffiliationGraphModel(random_state=42, **config["params"])
        G = agm.generate_graph()

        # Compute statistics
        stats = agm.compute_statistics()

        print(f"Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}")
        print(f"Density: {stats['density']:.4f}")
        print(f"Communities: {stats['n_communities']}")
        print(
            f"Avg community size: {stats['community_size_mean']:.1f} ± {stats['community_size_std']:.1f}"
        )
        print(f"Avg communities per node: {stats['avg_communities_per_node']:.2f}")
        print(
            f"Nodes in multiple communities: {stats['nodes_in_multiple_communities']}"
        )
        print(f"Clustering coefficient: {stats['clustering_coefficient']:.3f}")


if __name__ == "__main__":
    demo_agm()
