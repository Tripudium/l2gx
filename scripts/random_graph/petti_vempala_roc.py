"""
Petti-Vempala Random Overlapping Communities (ROC) Model

Implementation of the Random Overlapping Communities model from:
"Approximating Sparse Graphs: The Random Overlapping Communities Model"
by Samson Petti and Santosh Vempala

This model is specifically designed for theoretical analysis of community detection
algorithms and provides approximation guarantees for sparse graphs.

Key Properties of the Petti-Vempala ROC Model:
1. Each node belongs to exactly one or two communities
2. Communities are of roughly equal size
3. Edge probabilities follow specific theoretical framework
4. Designed for sparse graphs with controlled overlap structure

References:
- Petti, S., & Vempala, S. (2019). Approximating Sparse Graphs: The Random Overlapping Communities Model.
"""

from typing import Optional, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class PettiVempalaROC:
    """
    Implementation of the Petti-Vempala Random Overlapping Communities model.
    
    This model creates a sparse graph with overlapping community structure where:
    - Each node belongs to exactly 1 or 2 communities
    - Communities have roughly equal sizes
    - Edge probabilities are designed for theoretical analysis
    """
    
    def __init__(
        self,
        n_nodes: int,
        k_communities: int,
        p_in: float,
        p_out: float,
        overlap_fraction: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Petti-Vempala ROC model.
        
        Args:
            n_nodes: Number of nodes in the graph
            k_communities: Number of communities
            p_in: Probability of edge within a community
            p_out: Probability of edge between different communities
            overlap_fraction: Fraction of nodes that belong to two communities
            random_state: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.k_communities = k_communities
        self.p_in = p_in
        self.p_out = p_out
        self.overlap_fraction = overlap_fraction
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # Validate parameters
        if not 0 <= overlap_fraction <= 1:
            raise ValueError("overlap_fraction must be between 0 and 1")
        if not 0 <= p_in <= 1 or not 0 <= p_out <= 1:
            raise ValueError("Edge probabilities must be between 0 and 1")
        if p_in <= p_out:
            print("Warning: p_in should typically be larger than p_out for community structure")
            
        # Will be populated during generation
        self.node_communities = {}  # node -> list of community IDs
        self.community_nodes = defaultdict(set)  # community -> set of nodes
        self.graph = None
        
    def _assign_communities(self) -> None:
        """
        Assign communities to nodes according to the Petti-Vempala model.
        
        Each node belongs to exactly 1 or 2 communities:
        - (1 - overlap_fraction) of nodes belong to exactly 1 community
        - overlap_fraction of nodes belong to exactly 2 communities
        """
        n_overlap_nodes = int(self.overlap_fraction * self.n_nodes)
        n_single_nodes = self.n_nodes - n_overlap_nodes
        
        # Assign single-community nodes
        nodes = list(range(self.n_nodes))
        np.random.shuffle(nodes)
        
        # Distribute single-community nodes roughly equally across communities
        nodes_per_community = n_single_nodes // self.k_communities
        remainder = n_single_nodes % self.k_communities
        
        node_idx = 0
        
        # Assign nodes to single communities
        for community in range(self.k_communities):
            # Some communities get one extra node if there's a remainder
            community_size = nodes_per_community + (1 if community < remainder else 0)
            
            for _ in range(community_size):
                if node_idx < len(nodes):
                    node = nodes[node_idx]
                    self.node_communities[node] = [community]
                    self.community_nodes[community].add(node)
                    node_idx += 1
        
        # Assign overlapping nodes (belong to exactly 2 communities)
        for i in range(n_overlap_nodes):
            if node_idx < len(nodes):
                node = nodes[node_idx]
                
                # Choose 2 communities randomly
                communities = np.random.choice(
                    self.k_communities, size=2, replace=False
                )
                
                self.node_communities[node] = list(communities)
                for community in communities:
                    self.community_nodes[community].add(node)
                    
                node_idx += 1
    
    def _compute_edge_probability(self, node_i: int, node_j: int) -> float:
        """
        Compute edge probability between two nodes according to Petti-Vempala model.
        
        Edge probability depends on community overlap:
        - If nodes share at least one community: p_in
        - If nodes share no communities: p_out
        """
        communities_i = set(self.node_communities.get(node_i, []))
        communities_j = set(self.node_communities.get(node_j, []))
        
        # Check if nodes share any community
        if communities_i & communities_j:  # Intersection is non-empty
            return self.p_in
        else:
            return self.p_out
    
    def generate_graph(self) -> nx.Graph:
        """
        Generate a graph according to the Petti-Vempala ROC model.
        
        Returns:
            NetworkX graph with overlapping community structure
        """
        # Step 1: Assign communities
        self._assign_communities()
        
        # Step 2: Generate edges based on community structure
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        
        # Generate edges according to probabilities
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                edge_prob = self._compute_edge_probability(i, j)
                
                if np.random.random() < edge_prob:
                    G.add_edge(i, j)
        
        # Add community information as node attributes
        for node in G.nodes():
            G.nodes[node]['communities'] = self.node_communities.get(node, [])
            G.nodes[node]['n_communities'] = len(self.node_communities.get(node, []))
        
        self.graph = G
        return G
    
    def get_community_assignments(self) -> dict:
        """Get community assignments for each node."""
        return dict(self.node_communities)
    
    def get_community_nodes(self) -> dict:
        """Get nodes for each community."""
        return {k: list(v) for k, v in self.community_nodes.items()}
    
    def compute_statistics(self) -> dict:
        """Compute statistics about the generated graph and community structure."""
        if self.graph is None:
            raise ValueError("Must generate graph first")
            
        stats = {}
        
        # Basic graph statistics
        stats['n_nodes'] = self.graph.number_of_nodes()
        stats['n_edges'] = self.graph.number_of_edges()
        stats['density'] = nx.density(self.graph)
        stats['avg_degree'] = 2 * stats['n_edges'] / stats['n_nodes']
        
        # Community statistics
        stats['n_communities'] = self.k_communities
        community_sizes = [len(self.community_nodes[c]) for c in range(self.k_communities)]
        stats['community_size_mean'] = np.mean(community_sizes)
        stats['community_size_std'] = np.std(community_sizes)
        stats['community_size_min'] = np.min(community_sizes)
        stats['community_size_max'] = np.max(community_sizes)
        
        # Overlap statistics
        n_single_community = sum(1 for communities in self.node_communities.values() if len(communities) == 1)
        n_double_community = sum(1 for communities in self.node_communities.values() if len(communities) == 2)
        
        stats['nodes_single_community'] = n_single_community
        stats['nodes_double_community'] = n_double_community
        stats['actual_overlap_fraction'] = n_double_community / self.n_nodes
        stats['target_overlap_fraction'] = self.overlap_fraction
        
        # Edge type statistics
        intra_edges = 0  # Edges within communities
        inter_edges = 0  # Edges between communities
        
        for edge in self.graph.edges():
            node_i, node_j = edge
            communities_i = set(self.node_communities.get(node_i, []))
            communities_j = set(self.node_communities.get(node_j, []))
            
            if communities_i & communities_j:  # Share at least one community
                intra_edges += 1
            else:
                inter_edges += 1
        
        stats['intra_community_edges'] = intra_edges
        stats['inter_community_edges'] = inter_edges
        
        if stats['n_edges'] > 0:
            stats['intra_edge_fraction'] = intra_edges / stats['n_edges']
            stats['inter_edge_fraction'] = inter_edges / stats['n_edges']
        
        # Clustering coefficient
        stats['clustering_coefficient'] = nx.average_clustering(self.graph)
        
        # Theoretical expectations
        n_single = n_single_community
        n_double = n_double_community
        
        # Expected number of edges (approximately)
        # Within communities: roughly n_single * (n_single - 1) / 2 * p_in per community
        # Plus overlapping edges
        expected_intra = self.k_communities * (n_single // self.k_communities) * (n_single // self.k_communities - 1) / 2 * self.p_in
        expected_inter = (self.n_nodes * (self.n_nodes - 1) / 2 - expected_intra) * self.p_out
        
        stats['expected_intra_edges'] = expected_intra
        stats['expected_inter_edges'] = expected_inter
        stats['expected_total_edges'] = expected_intra + expected_inter
        
        return stats
    
    def visualize(
        self, 
        layout: str = "spring",
        node_size: int = 50,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize the generated graph showing community structure.
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
        
        # Plot 1: Color by number of communities (should be 1 or 2)
        node_colors = [len(self.node_communities.get(node, [])) for node in self.graph.nodes()]
        nx.draw(
            self.graph, pos, node_color=node_colors, node_size=node_size,
            with_labels=False, cmap='viridis', ax=axes[0, 0]
        )
        axes[0, 0].set_title("Nodes by # communities (1=blue, 2=yellow)")
        
        # Plot 2: Color by primary community
        primary_communities = []
        for node in self.graph.nodes():
            communities = self.node_communities.get(node, [])
            primary_communities.append(communities[0] if communities else -1)
        
        nx.draw(
            self.graph, pos, node_color=primary_communities, node_size=node_size,
            with_labels=False, cmap='Set1', ax=axes[0, 1]
        )
        axes[0, 1].set_title("Nodes by primary community")
        
        # Plot 3: Highlight overlapping nodes
        overlap_colors = ['red' if len(self.node_communities.get(node, [])) == 2 else 'lightblue' 
                         for node in self.graph.nodes()]
        nx.draw(
            self.graph, pos, node_color=overlap_colors, node_size=node_size,
            with_labels=False, ax=axes[1, 0]
        )
        axes[1, 0].set_title("Overlapping nodes (red)")
        
        # Plot 4: Show specific community
        if self.k_communities > 0:
            target_community = 0
            community_colors = []
            for node in self.graph.nodes():
                if target_community in self.node_communities.get(node, []):
                    community_colors.append('green')
                else:
                    community_colors.append('lightgray')
            
            nx.draw(
                self.graph, pos, node_color=community_colors, node_size=node_size,
                with_labels=False, ax=axes[1, 1]
            )
            axes[1, 1].set_title(f"Community {target_community} (green)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def demo_petti_vempala_roc():
    """Demonstrate the Petti-Vempala ROC model with different parameter settings."""
    print("PETTI-VEMPALA ROC MODEL DEMO")
    print("=" * 40)
    
    # Test different configurations following theoretical guidelines
    configs = [
        {
            "name": "Sparse with Low Overlap",
            "params": {
                "n_nodes": 100,
                "k_communities": 4,
                "p_in": 0.3,
                "p_out": 0.05,
                "overlap_fraction": 0.1,
            }
        },
        {
            "name": "Dense with High Overlap", 
            "params": {
                "n_nodes": 80,
                "k_communities": 3,
                "p_in": 0.5,
                "p_out": 0.1,
                "overlap_fraction": 0.3,
            }
        },
        {
            "name": "Many Communities",
            "params": {
                "n_nodes": 150,
                "k_communities": 6,
                "p_in": 0.25,
                "p_out": 0.02,
                "overlap_fraction": 0.15,
            }
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']}")
        print("-" * len(config['name']))
        
        # Generate graph
        roc = PettiVempalaROC(random_state=42, **config['params'])
        G = roc.generate_graph()
        
        # Compute statistics
        stats = roc.compute_statistics()
        
        print(f"Nodes: {stats['n_nodes']}, Edges: {stats['n_edges']}")
        print(f"Density: {stats['density']:.4f}, Avg degree: {stats['avg_degree']:.2f}")
        print(f"Communities: {stats['n_communities']}")
        print(f"Avg community size: {stats['community_size_mean']:.1f} ± {stats['community_size_std']:.1f}")
        print(f"Single-community nodes: {stats['nodes_single_community']}")
        print(f"Double-community nodes: {stats['nodes_double_community']}")
        print(f"Actual overlap fraction: {stats['actual_overlap_fraction']:.3f} (target: {stats['target_overlap_fraction']:.3f})")
        print(f"Intra-community edges: {stats['intra_community_edges']} ({stats.get('intra_edge_fraction', 0):.3f})")
        print(f"Inter-community edges: {stats['inter_community_edges']} ({stats.get('inter_edge_fraction', 0):.3f})")
        print(f"Clustering coefficient: {stats['clustering_coefficient']:.3f}")


if __name__ == "__main__":
    demo_petti_vempala_roc()