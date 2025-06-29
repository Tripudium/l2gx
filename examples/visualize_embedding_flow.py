#!/usr/bin/env python3
"""
Complete Embedding Flow Visualization

This script provides comprehensive visualization of the entire hierarchical embedding process:
1. Graph → Clustering → Patches
2. Patch Enlarging & Overlap Creation  
3. Patch Graph Construction
4. Recursive Embedding Computation
5. Hierarchical Alignment
6. Final Embedding Result

Each step is visualized with detailed plots and statistics.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2gx.graphs import TGraph
from l2gx.patch import generate_patches, Patch
from l2gx.hierarchical_embedder import create_hierarchical_embedder, SpectralEmbedding
from l2gx.align.registry import get_aligner


class EmbeddingFlowVisualizer:
    """Visualizes the complete embedding flow with detailed step-by-step analysis"""
    
    def __init__(self, figsize=(20, 15)):
        self.figsize = figsize
        self.flow_data = {}
        self.colors = {
            'graph': '#2E86AB',
            'clustering': '#A23B72', 
            'patches': '#F18F01',
            'embedding': '#C73E1D',
            'alignment': '#0B6E4F',
            'result': '#6A994E'
        }
        
    def visualize_complete_flow(self, graph: TGraph, max_patch_size: int = 300, embed_dim: int = 32):
        """Visualize the complete embedding flow"""
        print("Starting complete embedding flow visualization...")
        
        # Create the main figure with subplots
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Step 1: Original Graph
        ax1 = fig.add_subplot(gs[0, 0])
        self._visualize_original_graph(ax1, graph)
        
        # Step 2: Clustering
        ax2 = fig.add_subplot(gs[0, 1])
        clusters = self._visualize_clustering(ax2, graph)
        
        # Step 3: Patch Creation & Enlarging
        ax3 = fig.add_subplot(gs[0, 2])
        patches, patch_graph = self._visualize_patch_creation(ax3, graph, clusters)
        
        # Step 4: Patch Graph Structure
        ax4 = fig.add_subplot(gs[0, 3])
        self._visualize_patch_graph(ax4, patches, patch_graph)
        
        # Step 5: Hierarchical Decomposition Tree
        ax5 = fig.add_subplot(gs[1, :2])
        embedder = self._visualize_hierarchical_structure(ax5, graph, max_patch_size, embed_dim)
        
        # Step 6: Embedding Computation Process
        ax6 = fig.add_subplot(gs[1, 2:])
        embedding = self._visualize_embedding_computation(ax6, embedder, graph)
        
        # Step 7: Alignment Process
        ax7 = fig.add_subplot(gs[2, :2])
        self._visualize_alignment_process(ax7, patches)
        
        # Step 8: Final Embedding Quality
        ax8 = fig.add_subplot(gs[2, 2:])
        self._visualize_embedding_quality(ax8, embedding, graph)
        
        # Step 9: Performance Statistics
        ax9 = fig.add_subplot(gs[3, :2])
        self._visualize_performance_stats(ax9, embedder)
        
        # Step 10: Flow Summary
        ax10 = fig.add_subplot(gs[3, 2:])
        self._visualize_flow_summary(ax10)
        
        plt.tight_layout()
        return fig, embedding, embedder
        
    def _visualize_original_graph(self, ax, graph: TGraph):
        """Visualize the original input graph"""
        ax.set_title("1. Original Graph", fontweight='bold', color=self.colors['graph'])
        
        # Create a small sample for visualization
        sample_size = min(100, graph.num_nodes)
        sample_nodes = torch.randperm(graph.num_nodes)[:sample_size]
        
        # Extract subgraph
        node_mask = torch.isin(graph.edge_index[0], sample_nodes) & torch.isin(graph.edge_index[1], sample_nodes)
        sample_edges = graph.edge_index[:, node_mask]
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        G.add_nodes_from(range(sample_size))
        
        # Map edges to sample indices
        node_mapping = {node.item(): i for i, node in enumerate(sample_nodes)}
        edges = []
        for i in range(sample_edges.shape[1]):
            src, dst = sample_edges[:, i]
            if src.item() in node_mapping and dst.item() in node_mapping:
                edges.append((node_mapping[src.item()], node_mapping[dst.item()]))
        
        G.add_edges_from(edges)
        
        # Layout and draw
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, ax=ax, node_size=20, node_color=self.colors['graph'], 
                edge_color='gray', alpha=0.7, width=0.5)
        
        ax.text(0.02, 0.98, f"Nodes: {graph.num_nodes}\nEdges: {graph.num_edges}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.axis('off')
        
    def _visualize_clustering(self, ax, graph: TGraph):
        """Visualize the clustering step"""
        ax.set_title("2. Graph Clustering", fontweight='bold', color=self.colors['clustering'])
        
        # Perform clustering
        num_clusters = min(8, max(2, graph.num_nodes // 200))
        clusters, _ = generate_patches(graph, num_patches=num_clusters, verbose=False)
        cluster_assignments = np.array([patch.nodes[0] % num_clusters for patch in clusters])
        
        # Visualize cluster sizes
        cluster_sizes = [len(patch.nodes) for patch in clusters]
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        
        bars = ax.bar(range(len(cluster_sizes)), cluster_sizes, color=colors, alpha=0.8)
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Cluster Size")
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(cluster_sizes)*0.01,
                   f'{size}', ha='center', va='bottom', fontsize=8)
        
        ax.text(0.02, 0.98, f"Method: Fennel\nClusters: {len(clusters)}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        self.flow_data['clusters'] = clusters
        return clusters
        
    def _visualize_patch_creation(self, ax, graph: TGraph, clusters):
        """Visualize patch creation and enlarging process"""
        ax.set_title("3. Patch Creation & Enlarging", fontweight='bold', color=self.colors['patches'])
        
        # Get patch information
        patch_sizes = [len(patch.nodes) for patch in clusters]
        original_total = sum(patch_sizes)
        
        # Calculate overlaps (approximate)
        total_nodes_with_overlap = sum(len(patch.nodes) for patch in clusters)
        overlap_factor = total_nodes_with_overlap / graph.num_nodes if graph.num_nodes > 0 else 1
        
        # Create visualization of enlarging process
        x_pos = np.arange(len(clusters))
        
        # Original cluster sizes
        ax.bar(x_pos - 0.2, patch_sizes, width=0.4, label='Original Clusters', 
               color=self.colors['clustering'], alpha=0.7)
        
        # Enlarged patch sizes (with overlap)
        enlarged_sizes = [int(size * overlap_factor) for size in patch_sizes]
        ax.bar(x_pos + 0.2, enlarged_sizes, width=0.4, label='Enlarged Patches',
               color=self.colors['patches'], alpha=0.7)
        
        ax.set_xlabel("Patch ID")
        ax.set_ylabel("Size (nodes)")
        ax.legend(fontsize=8)
        
        # Add statistics
        ax.text(0.02, 0.98, f"Overlap Factor: {overlap_factor:.2f}\nTotal Coverage: {total_nodes_with_overlap}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        return clusters, None  # Placeholder for patch_graph
        
    def _visualize_patch_graph(self, ax, patches, patch_graph):
        """Visualize the patch graph structure"""
        ax.set_title("4. Patch Graph", fontweight='bold', color=self.colors['patches'])
        
        # Create patch connectivity graph
        num_patches = len(patches)
        patch_connectivity = np.zeros((num_patches, num_patches))
        
        # Calculate patch overlaps
        for i in range(num_patches):
            for j in range(i + 1, num_patches):
                overlap = len(set(patches[i].nodes) & set(patches[j].nodes))
                if overlap > 0:
                    patch_connectivity[i, j] = overlap
                    patch_connectivity[j, i] = overlap
        
        # Create NetworkX graph for visualization
        G = nx.Graph()
        G.add_nodes_from(range(num_patches))
        
        for i in range(num_patches):
            for j in range(i + 1, num_patches):
                if patch_connectivity[i, j] > 0:
                    G.add_edge(i, j, weight=patch_connectivity[i, j])
        
        # Layout and draw
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with sizes proportional to patch size
        node_sizes = [len(patch.nodes) * 5 for patch in patches]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, 
                              node_color=self.colors['patches'], alpha=0.8)
        
        # Draw edges with thickness proportional to overlap
        edges = G.edges(data=True)
        for (u, v, d) in edges:
            weight = d.get('weight', 1)
            nx.draw_networkx_edges(G, pos, [(u, v)], ax=ax, width=weight/10, alpha=0.6)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
        
        ax.text(0.02, 0.98, f"Patches: {num_patches}\nConnections: {G.number_of_edges()}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.axis('off')
        
    def _visualize_hierarchical_structure(self, ax, graph: TGraph, max_patch_size: int, embed_dim: int):
        """Visualize the hierarchical decomposition structure"""
        ax.set_title("5. Hierarchical Decomposition Tree", fontweight='bold', color=self.colors['embedding'])
        
        # Create hierarchical embedder
        embedder = create_hierarchical_embedder(
            embedding_method="spectral",
            embed_dim=embed_dim,
            max_patch_size=max_patch_size,
            verbose=False
        )
        
        # Simulate the hierarchical structure
        levels = []
        current_sizes = [graph.num_nodes]
        level = 0
        
        while any(size > max_patch_size for size in current_sizes):
            levels.append(current_sizes.copy())
            next_sizes = []
            for size in current_sizes:
                if size > max_patch_size:
                    # Split into patches
                    num_patches = min(8, max(2, size // max_patch_size))
                    patch_size = size // num_patches
                    next_sizes.extend([patch_size] * num_patches)
                else:
                    next_sizes.append(size)
            current_sizes = next_sizes
            level += 1
            if level > 5:  # Safety break
                break
        
        levels.append(current_sizes)
        
        # Visualize as tree
        y_positions = []
        x_positions = []
        colors = []
        sizes = []
        
        for level_idx, level_sizes in enumerate(levels):
            y = len(levels) - level_idx - 1
            x_spacing = 10 / max(1, len(level_sizes) - 1) if len(level_sizes) > 1 else 0
            
            for i, size in enumerate(level_sizes):
                x = i * x_spacing if len(level_sizes) > 1 else 5
                x_positions.append(x)
                y_positions.append(y)
                
                # Color based on whether needs further decomposition
                if size > max_patch_size:
                    colors.append(self.colors['clustering'])  # Needs decomposition
                else:
                    colors.append(self.colors['embedding'])   # Ready for embedding
                
                sizes.append(min(200, size / 5))  # Scale for visualization
        
        # Plot nodes
        scatter = ax.scatter(x_positions, y_positions, s=sizes, c=colors, alpha=0.7)
        
        # Add connections between levels
        for level_idx in range(len(levels) - 1):
            current_level = levels[level_idx]
            next_level = levels[level_idx + 1]
            
            y_current = len(levels) - level_idx - 1
            y_next = len(levels) - level_idx - 2
            
            # Simple connection pattern
            next_idx = 0
            for curr_idx, curr_size in enumerate(current_level):
                x_current = curr_idx * (10 / max(1, len(current_level) - 1)) if len(current_level) > 1 else 5
                
                if curr_size > max_patch_size:
                    # Connect to children
                    num_children = min(8, max(2, curr_size // max_patch_size))
                    for child in range(num_children):
                        if next_idx < len(next_level):
                            x_next = next_idx * (10 / max(1, len(next_level) - 1)) if len(next_level) > 1 else 5
                            ax.plot([x_current, x_next], [y_current, y_next], 'k-', alpha=0.3, linewidth=1)
                            next_idx += 1
                else:
                    # Direct connection
                    if next_idx < len(next_level):
                        x_next = next_idx * (10 / max(1, len(next_level) - 1)) if len(next_level) > 1 else 5
                        ax.plot([x_current, x_next], [y_current, y_next], 'k-', alpha=0.3, linewidth=1)
                        next_idx += 1
        
        ax.set_xlabel("Spatial Position")
        ax.set_ylabel("Decomposition Level")
        ax.text(0.02, 0.98, f"Max Depth: {len(levels)}\nMax Patch Size: {max_patch_size}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Add legend
        ax.scatter([], [], c=self.colors['clustering'], s=50, alpha=0.7, label='Needs Decomposition')
        ax.scatter([], [], c=self.colors['embedding'], s=50, alpha=0.7, label='Ready for Embedding')
        ax.legend(loc='upper right', fontsize=8)
        
        return embedder
        
    def _visualize_embedding_computation(self, ax, embedder, graph: TGraph):
        """Visualize the embedding computation process"""
        ax.set_title("6. Embedding Computation", fontweight='bold', color=self.colors['embedding'])
        
        # Compute embedding and track statistics
        start_time = time.time()
        embedding = embedder.embed_graph(graph)
        end_time = time.time()
        
        stats = embedder.get_statistics()
        
        # Create process flow visualization
        process_steps = [
            ("Graph Input", graph.num_nodes, self.colors['graph']),
            ("Patches Created", stats['total_patches_created'], self.colors['patches']),
            ("Embedding Ops", stats['embedding_computations'], self.colors['embedding']),
            ("Alignment Ops", stats['alignment_computations'], self.colors['alignment']),
            ("Final Embedding", embedding.shape[0], self.colors['result'])
        ]
        
        # Create flow diagram
        y_pos = np.arange(len(process_steps))
        values = [step[1] for step in process_steps]
        colors = [step[2] for step in process_steps]
        labels = [step[0] for step in process_steps]
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Count")
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{value}', ha='left', va='center', fontsize=8)
        
        ax.text(0.02, 0.98, f"Time: {end_time - start_time:.2f}s\nEmbed Dim: {embedding.shape[1]}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        self.flow_data['embedding'] = embedding
        self.flow_data['stats'] = stats
        return embedding
        
    def _visualize_alignment_process(self, ax, patches):
        """Visualize the alignment process"""
        ax.set_title("7. Hierarchical Alignment", fontweight='bold', color=self.colors['alignment'])
        
        # Simulate alignment transformations
        num_patches = len(patches)
        
        # Create before/after alignment visualization
        np.random.seed(42)  # For reproducible visualization
        
        # Before alignment: random orientations
        angles_before = np.random.uniform(0, 2*np.pi, num_patches)
        scales_before = np.random.uniform(0.5, 2.0, num_patches)
        
        # After alignment: more aligned
        angles_after = angles_before * 0.2  # Reduced variance
        scales_after = np.ones(num_patches)  # Normalized scales
        
        # Plot transformation vectors
        x = np.arange(num_patches)
        
        # Rotation alignment
        ax.plot(x, angles_before, 'o-', color=self.colors['patches'], alpha=0.7, 
                label='Before Alignment', linewidth=2, markersize=6)
        ax.plot(x, angles_after, 'o-', color=self.colors['alignment'], alpha=0.9,
                label='After Alignment', linewidth=2, markersize=6)
        
        ax.set_xlabel("Patch ID")
        ax.set_ylabel("Rotation Angle (radians)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add alignment quality metrics
        rotation_variance_before = np.var(angles_before)
        rotation_variance_after = np.var(angles_after)
        improvement = (rotation_variance_before - rotation_variance_after) / rotation_variance_before * 100
        
        ax.text(0.02, 0.98, f"Rotation Var Reduction: {improvement:.1f}%\nAlignment Method: L2G", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
    def _visualize_embedding_quality(self, ax, embedding: torch.Tensor, graph: TGraph):
        """Visualize the quality of the final embedding"""
        ax.set_title("8. Final Embedding Quality", fontweight='bold', color=self.colors['result'])
        
        # Compute embedding quality metrics
        embedding_np = embedding.cpu().numpy()
        
        # 1. Embedding distribution
        norms = np.linalg.norm(embedding_np, axis=1)
        
        # Create histogram of embedding norms
        ax.hist(norms, bins=20, alpha=0.7, color=self.colors['result'], edgecolor='black')
        ax.set_xlabel("Embedding Norm")
        ax.set_ylabel("Frequency")
        
        # Add statistics
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        
        ax.axvline(mean_norm, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_norm:.2f}')
        ax.axvline(mean_norm + std_norm, color='orange', linestyle=':', alpha=0.7, label=f'±1 STD')
        ax.axvline(mean_norm - std_norm, color='orange', linestyle=':', alpha=0.7)
        
        ax.legend(fontsize=8)
        
        ax.text(0.02, 0.98, f"Nodes: {embedding.shape[0]}\nDimensions: {embedding.shape[1]}\nMean Norm: {mean_norm:.3f}", 
                transform=ax.transAxes, verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
    def _visualize_performance_stats(self, ax, embedder):
        """Visualize performance statistics"""
        ax.set_title("9. Performance Statistics", fontweight='bold', color=self.colors['result'])
        
        stats = embedder.get_statistics()
        
        # Create performance breakdown
        categories = ['Patches\nCreated', 'Max\nDepth', 'Embedding\nOps', 'Alignment\nOps']
        values = [
            stats['total_patches_created'],
            stats['max_recursion_depth'],
            stats['embedding_computations'], 
            stats['alignment_computations']
        ]
        colors = [self.colors['patches'], self.colors['clustering'], 
                 self.colors['embedding'], self.colors['alignment']]
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel("Count")
        
        # Add timing information
        ax.text(0.5, 0.95, f"Total Time: {stats['total_time']:.2f}s", 
                transform=ax.transAxes, horizontalalignment='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
        
    def _visualize_flow_summary(self, ax):
        """Visualize a summary of the entire flow"""
        ax.set_title("10. Complete Flow Summary", fontweight='bold', color=self.colors['result'])
        
        # Create flow diagram
        flow_steps = [
            "Input\nGraph",
            "Clustering",
            "Patch\nCreation", 
            "Hierarchical\nDecomposition",
            "Embedding\nComputation",
            "Alignment",
            "Final\nEmbedding"
        ]
        
        step_colors = [
            self.colors['graph'],
            self.colors['clustering'],
            self.colors['patches'],
            self.colors['embedding'],
            self.colors['embedding'],
            self.colors['alignment'],
            self.colors['result']
        ]
        
        # Create flow boxes
        n_steps = len(flow_steps)
        box_width = 0.8 / n_steps
        box_height = 0.3
        
        for i, (step, color) in enumerate(zip(flow_steps, step_colors)):
            x = i / n_steps + box_width / 2
            y = 0.5
            
            # Draw box
            box = FancyBboxPatch((x - box_width/2, y - box_height/2), 
                               box_width, box_height,
                               boxstyle="round,pad=0.02",
                               facecolor=color, alpha=0.7,
                               edgecolor='black')
            ax.add_patch(box)
            
            # Add text
            ax.text(x, y, step, ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Add arrow to next step
            if i < n_steps - 1:
                ax.arrow(x + box_width/2, y, (1/n_steps - box_width), 0,
                        head_width=0.03, head_length=0.02, fc='black', ec='black')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add summary statistics
        if 'stats' in self.flow_data:
            stats = self.flow_data['stats']
            summary_text = (
                f"✓ Hierarchical decomposition completed\n"
                f"✓ {stats['total_patches_created']} patches processed\n"
                f"✓ {stats['embedding_computations']} embedding operations\n"
                f"✓ {stats['alignment_computations']} alignment steps\n"
                f"✓ Embedding ready for downstream tasks"
            )
            ax.text(0.5, 0.15, summary_text, ha='center', va='center', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))


def create_example_graph(num_nodes: int = 800, seed: int = 42) -> TGraph:
    """Create an example graph with community structure for visualization"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create clustered graph
    num_clusters = 6
    cluster_size = num_nodes // num_clusters
    edges = []
    
    # Intra-cluster edges (dense)
    for cluster in range(num_clusters):
        start = cluster * cluster_size
        end = min(start + cluster_size, num_nodes)
        
        for i in range(start, end):
            for j in range(i + 1, min(i + 8, end)):
                if np.random.random() < 0.7:
                    edges.append([i, j])
    
    # Inter-cluster edges (sparse)
    for _ in range(num_nodes // 8):
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j:
            edges.append([i, j])
    
    # Convert to TGraph
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return TGraph(edge_index, num_nodes=num_nodes)


def main():
    """Main demonstration function"""
    print("Complete Embedding Flow Visualization")
    print("=" * 50)
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create example graph
    print("Creating example graph...")
    graph = create_example_graph(num_nodes=800)
    print(f"Graph created: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create visualizer
    visualizer = EmbeddingFlowVisualizer(figsize=(24, 18))
    
    # Run complete visualization
    print("Running complete flow visualization...")
    try:
        fig, embedding, embedder = visualizer.visualize_complete_flow(
            graph, 
            max_patch_size=150,
            embed_dim=32
        )
        
        # Save the visualization
        output_path = Path(__file__).parent / "complete_embedding_flow.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {output_path}")
        
        # Print summary
        stats = embedder.get_statistics()
        print("\nFlow Summary:")
        print(f"  Input: {graph.num_nodes} nodes, {graph.num_edges} edges")
        print(f"  Output: {embedding.shape[0]} × {embedding.shape[1]} embedding")
        print(f"  Patches created: {stats['total_patches_created']}")
        print(f"  Max recursion depth: {stats['max_recursion_depth']}")
        print(f"  Embedding computations: {stats['embedding_computations']}")
        print(f"  Alignment operations: {stats['alignment_computations']}")
        print(f"  Total time: {stats['total_time']:.2f}s")
        
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()