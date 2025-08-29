"""
General Overlapping Communities Model

This model generates graphs by creating overlapping communities through various 
random processes, then connecting nodes based on their community structure.
This is a flexible, general-purpose model for testing patch-based algorithms.

Key features:
1. Flexible community generation (planted, grown, preferential, or hybrid)
2. Nodes can belong to multiple communities with varying membership strengths
3. Edge probabilities can depend on membership strengths or just membership
4. Supports community evolution and growth patterns

Note: This is NOT the specific Petti-Vempala ROC model. For that model, 
see petti_vempala_roc.py.

References:
- Palla, G., Derényi, I., Farkas, I., & Vicsek, T. (2005). Uncovering the overlapping community structure of complex networks. Nature.
- Fortunato, S., & Hric, D. (2016). Community detection in networks: A user guide. Physics Reports.
- Granell, C., Darst, R. K., Arenas, A., Fortunato, S., & Gómez, S. (2015). Benchmark model for overlapping community structure. Physical Review E.
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class GeneralOverlappingCommunities:
    """
    General Overlapping Communities model for generating realistic overlapping community structure.

    This model creates communities through various growth processes and then generates
    edges based on community membership and membership strengths. This is a flexible,
    general-purpose model, not the specific Petti-Vempala ROC model.
    """

    def __init__(
        self,
        n_nodes: int,
        n_communities: int,
        community_generation: str = "grown",
        average_community_size: float = 20,
        community_size_variance: float = 0.5,
        overlap_factor: float = 0.3,
        membership_strength_dist: str = "binary",
        edge_prob_function: str = "strength_product",
        base_edge_prob: float = 0.1,
        strength_exponent: float = 2.0,
        background_edge_prob: float = 0.005,
        random_state: int | None = None,
    ):
        """
        Initialize the ROC model.

        Args:
            n_nodes: Number of nodes in the graph
            n_communities: Number of communities to generate
            community_generation: How to generate communities
                                 ("planted", "grown", "preferential", "hybrid")
            average_community_size: Expected size of communities
            community_size_variance: Variance in community sizes (0-1)
            overlap_factor: Controls amount of overlap between communities
            membership_strength_dist: Distribution of membership strengths
                                    ("binary", "uniform", "beta", "exponential")
            edge_prob_function: How to compute edge probability from memberships
                              ("strength_product", "max_strength", "threshold", "noisy_or")
            base_edge_prob: Base edge probability scaling factor
            strength_exponent: Exponent for strength-based edge probabilities
            background_edge_prob: Probability of random background edges
            random_state: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.community_generation = community_generation
        self.average_community_size = average_community_size
        self.community_size_variance = community_size_variance
        self.overlap_factor = overlap_factor
        self.membership_strength_dist = membership_strength_dist
        self.edge_prob_function = edge_prob_function
        self.base_edge_prob = base_edge_prob
        self.strength_exponent = strength_exponent
        self.background_edge_prob = background_edge_prob

        if random_state is not None:
            np.random.seed(random_state)

        # Will be populated during generation
        self.node_communities = {}  # node -> {community: strength}
        self.community_nodes = defaultdict(dict)  # community -> {node: strength}
        self.graph = None

    def _generate_community_sizes(self) -> list[int]:
        """Generate sizes for each community."""
        sizes = []

        if self.community_size_variance == 0:
            # Fixed size communities
            sizes = [int(self.average_community_size)] * self.n_communities
        else:
            # Variable size communities
            for _ in range(self.n_communities):
                # Use gamma distribution for positive, skewed sizes
                shape = 1.0 / self.community_size_variance
                scale = self.average_community_size / shape
                size = max(1, int(np.random.gamma(shape, scale)))
                size = min(size, self.n_nodes)  # Cap at total nodes
                sizes.append(size)

        return sizes

    def _generate_membership_strength(self) -> float:
        """Generate a membership strength value."""
        if self.membership_strength_dist == "binary":
            return 1.0
        elif self.membership_strength_dist == "uniform":
            return np.random.uniform(0.1, 1.0)
        elif self.membership_strength_dist == "beta":
            # Beta distribution skewed toward higher values
            return np.random.beta(2, 1)
        elif self.membership_strength_dist == "exponential":
            # Exponential with mean 0.5, clipped to [0.1, 1.0]
            strength = np.random.exponential(0.5)
            return np.clip(strength, 0.1, 1.0)
        else:
            raise ValueError(
                f"Unknown membership strength distribution: {self.membership_strength_dist}"
            )

    def _generate_planted_communities(self, sizes: list[int]) -> None:
        """Generate communities by randomly planting them."""
        for c, size in enumerate(sizes):
            # Select random nodes for this community
            nodes = np.random.choice(self.n_nodes, size=size, replace=False)

            for node in nodes:
                strength = self._generate_membership_strength()

                if node not in self.node_communities:
                    self.node_communities[node] = {}
                self.node_communities[node][c] = strength
                self.community_nodes[c][node] = strength

    def _generate_grown_communities(self, sizes: list[int]) -> None:
        """Generate communities by growing them from seed nodes."""
        for c, target_size in enumerate(sizes):
            # Start with a random seed node
            seed = np.random.randint(self.n_nodes)
            community_nodes = {seed}

            if seed not in self.node_communities:
                self.node_communities[seed] = {}
            self.node_communities[seed][c] = self._generate_membership_strength()
            self.community_nodes[c][seed] = self.node_communities[seed][c]

            # Grow community using preferential attachment within neighborhoods
            candidates = set(range(self.n_nodes)) - community_nodes

            while len(community_nodes) < target_size and candidates:
                # Choose next node based on proximity to existing community members
                # For now, use simple random selection (could implement spatial growth)

                if (
                    np.random.random() < self.overlap_factor
                    and len(community_nodes) > 2
                ):
                    # Occasionally add nodes that are already in other communities (overlap)
                    overlapping_candidates = [
                        n for n in candidates if n in self.node_communities
                    ]
                    if overlapping_candidates:
                        next_node = np.random.choice(overlapping_candidates)
                    else:
                        next_node = np.random.choice(list(candidates))
                else:
                    # Add a completely new node
                    next_node = np.random.choice(list(candidates))

                community_nodes.add(next_node)
                candidates.remove(next_node)

                strength = self._generate_membership_strength()
                if next_node not in self.node_communities:
                    self.node_communities[next_node] = {}
                self.node_communities[next_node][c] = strength
                self.community_nodes[c][next_node] = strength

    def _generate_preferential_communities(self, sizes: list[int]) -> None:
        """Generate communities using preferential attachment."""
        if not sizes:
            return
            
        # Get the next available community IDs
        existing_communities = max(self.community_nodes.keys()) if self.community_nodes else -1
        start_community_id = existing_communities + 1
        
        # First create small seed communities
        for i, size in enumerate(sizes):
            community_id = start_community_id + i
            # Start with 2-3 seed nodes
            seed_size = min(3, size)
            seeds = np.random.choice(self.n_nodes, size=seed_size, replace=False)

            for node in seeds:
                strength = self._generate_membership_strength()
                if node not in self.node_communities:
                    self.node_communities[node] = {}
                self.node_communities[node][community_id] = strength
                self.community_nodes[community_id][node] = strength

        # Grow communities preferentially
        for i, target_size in enumerate(sizes):
            community_id = start_community_id + i
            current_size = len(self.community_nodes[community_id])

            while current_size < target_size:
                # Choose new node preferentially based on number of communities it's in
                candidates = [
                    n for n in range(self.n_nodes) if n not in self.community_nodes[community_id]
                ]
                if not candidates:
                    break

                # Compute selection probabilities (prefer nodes with fewer memberships)
                probs = []
                for node in candidates:
                    current_memberships = len(self.node_communities.get(node, {}))
                    # Prefer nodes with fewer current memberships (with some randomness)
                    prob = 1.0 / (1.0 + current_memberships * (1 - self.overlap_factor))
                    probs.append(prob)

                probs = np.array(probs)
                probs = probs / probs.sum()

                # Select node
                selected_idx = np.random.choice(len(candidates), p=probs)
                selected_node = candidates[selected_idx]

                strength = self._generate_membership_strength()
                if selected_node not in self.node_communities:
                    self.node_communities[selected_node] = {}
                self.node_communities[selected_node][community_id] = strength
                self.community_nodes[community_id][selected_node] = strength

                current_size += 1

    def _generate_communities(self) -> None:
        """Generate communities according to the specified method."""
        sizes = self._generate_community_sizes()

        if self.community_generation == "planted":
            self._generate_planted_communities(sizes)
        elif self.community_generation == "grown":
            self._generate_grown_communities(sizes)
        elif self.community_generation == "preferential":
            self._generate_preferential_communities(sizes)
        elif self.community_generation == "hybrid":
            # Mix of different generation methods
            n_planted = self.n_communities // 3
            n_grown = self.n_communities // 3
            n_preferential = self.n_communities - n_planted - n_grown

            if n_planted > 0:
                self._generate_planted_communities(sizes[:n_planted])
            if n_grown > 0:
                grown_start = n_planted
                grown_end = n_planted + n_grown
                self._generate_grown_communities(sizes[grown_start:grown_end])
            if n_preferential > 0:
                pref_start = n_planted + n_grown
                self._generate_preferential_communities(sizes[pref_start:])
        else:
            raise ValueError(
                f"Unknown community generation method: {self.community_generation}"
            )

        # Ensure every node is in at least one community
        for node in range(self.n_nodes):
            if node not in self.node_communities or not self.node_communities[node]:
                # Assign to random community
                c = np.random.randint(self.n_communities)
                strength = self._generate_membership_strength()
                if node not in self.node_communities:
                    self.node_communities[node] = {}
                self.node_communities[node][c] = strength
                self.community_nodes[c][node] = strength

    def _compute_edge_probability(self, node_i: int, node_j: int) -> float:
        """Compute edge probability between two nodes based on their community memberships."""
        communities_i = self.node_communities.get(node_i, {})
        communities_j = self.node_communities.get(node_j, {})

        if not communities_i or not communities_j:
            return self.background_edge_prob

        # Find shared communities and compute combined probability
        shared_communities = set(communities_i.keys()) & set(communities_j.keys())

        if not shared_communities:
            return self.background_edge_prob

        # Compute probability based on specified function
        if self.edge_prob_function == "strength_product":
            # Product of membership strengths in shared communities
            prob = 0.0
            for c in shared_communities:
                strength_product = communities_i[c] * communities_j[c]
                prob += self.base_edge_prob * (strength_product**self.strength_exponent)
            prob = min(prob, 1.0)  # Cap at 1.0

        elif self.edge_prob_function == "max_strength":
            # Maximum strength across shared communities
            max_strength = 0.0
            for c in shared_communities:
                strength_product = communities_i[c] * communities_j[c]
                max_strength = max(max_strength, strength_product)
            prob = self.base_edge_prob * (max_strength**self.strength_exponent)

        elif self.edge_prob_function == "threshold":
            # Binary threshold based on strength
            threshold = 0.5
            prob = 0.0
            for c in shared_communities:
                if communities_i[c] >= threshold and communities_j[c] >= threshold:
                    prob = max(prob, self.base_edge_prob)

        elif self.edge_prob_function == "noisy_or":
            # Noisy-OR combination over shared communities
            prob_no_edge = 1.0
            for c in shared_communities:
                strength_product = communities_i[c] * communities_j[c]
                community_prob = self.base_edge_prob * (
                    strength_product**self.strength_exponent
                )
                prob_no_edge *= 1.0 - community_prob
            prob = 1.0 - prob_no_edge

        else:
            raise ValueError(
                f"Unknown edge probability function: {self.edge_prob_function}"
            )

        return prob

    def generate_graph(self) -> nx.Graph:
        """
        Generate a graph using the ROC model.

        Returns:
            NetworkX graph with overlapping community structure
        """
        # Step 1: Generate communities
        self._generate_communities()

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
            G.nodes[node]["communities"] = list(
                self.node_communities.get(node, {}).keys()
            )
            G.nodes[node]["community_strengths"] = dict(
                self.node_communities.get(node, {})
            )

        self.graph = G
        return G

    def get_community_memberships(self) -> dict[int, dict[int, float]]:
        """Get community memberships with strengths for each node."""
        return dict(self.node_communities)

    def get_community_nodes(self) -> dict[int, dict[int, float]]:
        """Get nodes with strengths for each community."""
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
            len(self.node_communities.get(node, {})) for node in range(self.n_nodes)
        ]
        stats["avg_communities_per_node"] = np.mean(node_memberships)
        stats["max_communities_per_node"] = np.max(node_memberships)
        stats["nodes_in_multiple_communities"] = sum(
            1 for m in node_memberships if m > 1
        )
        stats["overlap_ratio"] = stats["nodes_in_multiple_communities"] / self.n_nodes

        # Membership strength statistics
        all_strengths = []
        for node_comms in self.node_communities.values():
            all_strengths.extend(node_comms.values())

        if all_strengths:
            stats["avg_membership_strength"] = np.mean(all_strengths)
            stats["min_membership_strength"] = np.min(all_strengths)
            stats["max_membership_strength"] = np.max(all_strengths)

        # Clustering coefficient
        stats["clustering_coefficient"] = nx.average_clustering(self.graph)

        return stats

    def visualize(
        self,
        layout: str = "spring",
        node_size: int = 50,
        figsize: tuple[int, int] = (16, 8),
        save_path: str | None = None,
    ) -> None:
        """
        Visualize the generated graph showing community structure and membership strengths.
        """
        if self.graph is None:
            raise ValueError("Must generate graph first")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(self.graph, k=0.5, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph)

        # Plot 1: Color by number of communities
        node_colors = [
            len(self.node_communities.get(node, {})) for node in self.graph.nodes()
        ]
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            with_labels=False,
            cmap="viridis",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Nodes by # communities")

        # Plot 2: Color by maximum membership strength
        max_strengths = []
        for node in self.graph.nodes():
            strengths = list(self.node_communities.get(node, {}).values())
            max_strengths.append(max(strengths) if strengths else 0)

        nx.draw(
            self.graph,
            pos,
            node_color=max_strengths,
            node_size=node_size,
            with_labels=False,
            cmap="plasma",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Nodes by max membership strength")

        # Plot 3: Highlight specific community
        if self.n_communities > 0:
            target_community = 0
            node_colors = []
            for node in self.graph.nodes():
                if target_community in self.node_communities.get(node, {}):
                    strength = self.node_communities[node][target_community]
                    node_colors.append(strength)
                else:
                    node_colors.append(0)

            nx.draw(
                self.graph,
                pos,
                node_color=node_colors,
                node_size=node_size,
                with_labels=False,
                cmap="Reds",
                ax=axes[1, 0],
            )
            axes[1, 0].set_title(f"Community {target_community} membership strength")

        # Plot 4: Overlapping nodes
        node_colors = [
            "orange" if len(self.node_communities.get(node, {})) > 1 else "lightblue"
            for node in self.graph.nodes()
        ]
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            with_labels=False,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Overlapping nodes (orange)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


def demo_general_overlapping():
    """Demonstrate the General Overlapping Communities model with different parameter settings."""
    print("GENERAL OVERLAPPING COMMUNITIES MODEL DEMO")
    print("=" * 50)

    # Test different configurations
    configs = [
        {
            "name": "Grown Communities",
            "params": {
                "n_nodes": 120,
                "n_communities": 6,
                "community_generation": "grown",
                "average_community_size": 25,
                "overlap_factor": 0.2,
                "membership_strength_dist": "uniform",
                "base_edge_prob": 0.15,
            },
        },
        {
            "name": "Preferential Communities",
            "params": {
                "n_nodes": 100,
                "n_communities": 5,
                "community_generation": "preferential",
                "average_community_size": 20,
                "overlap_factor": 0.4,
                "membership_strength_dist": "beta",
                "edge_prob_function": "strength_product",
                "base_edge_prob": 0.2,
            },
        },
        {
            "name": "High Strength Variance",
            "params": {
                "n_nodes": 80,
                "n_communities": 4,
                "community_generation": "hybrid",
                "average_community_size": 25,
                "community_size_variance": 0.8,
                "membership_strength_dist": "exponential",
                "edge_prob_function": "noisy_or",
                "base_edge_prob": 0.1,
            },
        },
    ]

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * len(config["name"]))

        # Generate graph
        roc = GeneralOverlappingCommunities(random_state=42, **config["params"])
        G = roc.generate_graph()

        # Compute statistics
        stats = roc.compute_statistics()

        print(f"Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}")
        print(f"Density: {stats['density']:.4f}")
        print(f"Communities: {stats['n_communities']}")
        print(
            f"Avg community size: {stats['community_size_mean']:.1f} ± {stats['community_size_std']:.1f}"
        )
        print(f"Avg communities per node: {stats['avg_communities_per_node']:.2f}")
        print(f"Overlap ratio: {stats['overlap_ratio']:.2f}")
        if "avg_membership_strength" in stats:
            print(f"Avg membership strength: {stats['avg_membership_strength']:.2f}")
        print(f"Clustering coefficient: {stats['clustering_coefficient']:.3f}")


if __name__ == "__main__":
    demo_general_overlapping()
