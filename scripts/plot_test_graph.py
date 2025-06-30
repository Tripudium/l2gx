#!/usr/bin/env python3
"""
Plot Test Graph

This script generates and visualizes a test graph using the generate_test_graph
function and NetworkX plotting utilities.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
import sys
import argparse

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.test_graph_utils import generate_hidden_partition_model

def tgraph_to_networkx(tgraph):
    """Convert TGraph to NetworkX graph"""
    G = nx.Graph()
    G.add_nodes_from(range(tgraph.num_nodes))
    
    # Convert edge_index to edge list
    edge_index = tgraph.edge_index.cpu().numpy()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i])) 
             for i in range(edge_index.shape[1])]
    
    G.add_edges_from(edges)
    return G


def create_cluster_aware_layout(G, cluster_assignments):
    """Create a layout that groups nodes from the same cluster together"""
    
    # Assign nodes to clusters based on node IDs (matching the generation pattern)
    clusters = {}
    for node in G.nodes():
        cluster_id = cluster_assignments[node]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)
    
    # Create cluster positions in a circular arrangement
    cluster_centers = {}
    angle_step = 2 * np.pi / len(clusters)
    radius = 3.0  # Distance between cluster centers
    
    for i, cluster_id in enumerate(sorted(clusters.keys())):
        angle = i * angle_step
        cluster_centers[cluster_id] = (radius * np.cos(angle), radius * np.sin(angle))
    
    # Position nodes within each cluster
    pos = {}
    for cluster_id, nodes in clusters.items():
        center_x, center_y = cluster_centers[cluster_id]
        
        if len(nodes) == 1:
            pos[nodes[0]] = (center_x, center_y)
        else:
            # Create subgraph for this cluster
            subgraph = G.subgraph(nodes)
            
            # Use spring layout for nodes within the cluster
            try:
                sub_pos = nx.spring_layout(subgraph, k=0.3, iterations=50, 
                                         center=(center_x, center_y), scale=0.8)
                pos.update(sub_pos)
            except Exception:
                # Fallback: arrange in a small circle around cluster center
                for j, node in enumerate(nodes):
                    sub_angle = (2 * np.pi * j) / len(nodes)
                    sub_radius = 0.5
                    pos[node] = (center_x + sub_radius * np.cos(sub_angle),
                               center_y + sub_radius * np.sin(sub_angle))
    
    return pos


def plot_graph_with_clusters(G, cluster_assignments, figsize=(12, 10)):
    """Plot graph with cluster-aware layout and coloring"""
    plt.figure(figsize=figsize)
    
    # Create cluster assignments based on node IDs (approximating the clustering structure)
    num_clusters = len(set(cluster_assignments))
    node_colors = []
    cluster_colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))
    
    for node in G.nodes():
        cluster_id = cluster_assignments[node]
        node_colors.append(cluster_colors[cluster_id])
    
    # Use cluster-aware layout
    pos = create_cluster_aware_layout(G, cluster_assignments)
    
    # Draw the graph
    nx.draw(G, pos, 
            node_color=node_colors,
            node_size=60,
            edge_color='gray',
            alpha=0.8,
            width=0.5)
    
    plt.title(f"Test Graph with Cluster Layout ({len(G.nodes)} nodes, {len(G.edges)} edges)")
    plt.axis('off')
    return plt.gcf()


def plot_graph_statistics(G):
    """Plot basic graph statistics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    ax1.hist(degrees, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Degree Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Clustering coefficient distribution
    clustering_coeffs = list(nx.clustering(G).values())
    ax2.hist(clustering_coeffs, bins=20, alpha=0.7, edgecolor='black', color='orange')
    ax2.set_xlabel('Clustering Coefficient')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Clustering Coefficient Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Connected components
    components = list(nx.connected_components(G))
    component_sizes = [len(c) for c in components]
    ax3.bar(range(len(component_sizes)), sorted(component_sizes, reverse=True), 
            alpha=0.7, color='green')
    ax3.set_xlabel('Component Rank')
    ax3.set_ylabel('Component Size')
    ax3.set_title('Connected Component Sizes')
    ax3.grid(True, alpha=0.3)
    
    # Graph statistics text
    stats_text = f"""Graph Statistics:
Nodes: {len(G.nodes)}
Edges: {len(G.edges)}
Avg Degree: {np.mean(degrees):.2f}
Density: {nx.density(G):.4f}
Avg Clustering: {np.mean(clustering_coeffs):.3f}
Components: {len(components)}
Largest Component: {max(component_sizes) if component_sizes else 0}
"""
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace')
    ax4.axis('off')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate and plot test graph")
    parser.add_argument("--nodes", type=int, default=100, help="Number of nodes")
    parser.add_argument("--degree", type=int, default=10, help="Average degree")
    parser.add_argument("--clusters", type=int, default=10, help="Number of clusters for coloring")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save", action="store_true", help="Save plots to files")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    
    args = parser.parse_args()
    
    print(f"Generating test graph with {args.nodes} nodes, avg degree {args.degree}...")
    
    # Generate test graph
    tgraph, cluster_assignments = generate_hidden_partition_model(args.nodes, args.clusters, seed=args.seed)
    print(f"Generated TGraph: {tgraph.num_nodes} nodes, {tgraph.num_edges} edges")
    
    # Convert to NetworkX
    G = tgraph_to_networkx(tgraph)
    print(f"Converted to NetworkX: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Plot the graph
    print("Creating graph visualization...")
    fig1 = plot_graph_with_clusters(G, cluster_assignments)
    
    # Plot statistics
    #print("Creating statistics plots...")
    #fig2 = plot_graph_statistics(G)
    
    # Save plots if requested
    if args.save:
        output_dir = Path(__file__).parent.parent / "plots"
        output_dir.mkdir(exist_ok=True)
        
        graph_plot_path = output_dir / f"test_graph_{args.nodes}nodes.png"
        #stats_plot_path = output_dir / f"test_graph_stats_{args.nodes}nodes.png"
        
        fig1.savefig(graph_plot_path, dpi=300, bbox_inches='tight')
        #fig2.savefig(stats_plot_path, dpi=300, bbox_inches='tight')
        
        print(f"Saved graph plot to: {graph_plot_path}")
        #print(f"Saved statistics plot to: {stats_plot_path}")
    
    # Show plots unless disabled
    if not args.no_show:
        plt.show()
    else:
        plt.close('all')


if __name__ == "__main__":
    main()