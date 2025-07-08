#!/usr/bin/env python3
"""
Hierarchical Embedding Demonstration

This script demonstrates the hierarchical embedding approach that balances
computational complexity between embedding computation and graph decomposition.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2gx.graphs import TGraph
from l2gx.hierarchical_embedder import (
    HierarchicalEmbedder,
    SpectralEmbedding,
    RandomWalkEmbedding,
    create_hierarchical_embedder
)


def create_hierarchical_graph(num_nodes: int = 3000, num_levels: int = 3, seed: int = 42) -> TGraph:
    """Create a graph with hierarchical community structure"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    edges = []
    nodes_per_level = num_nodes // (2 ** num_levels)
    
    # Build hierarchical structure
    current_start = 0
    
    for level in range(num_levels):
        cluster_size = nodes_per_level * (2 ** level)
        num_clusters = 2 ** (num_levels - level - 1)
        
        for cluster in range(num_clusters):
            cluster_start = current_start + cluster * cluster_size
            cluster_end = min(cluster_start + cluster_size, num_nodes)
            
            # Dense connections within cluster
            for i in range(cluster_start, cluster_end):
                for j in range(i + 1, min(i + 15, cluster_end)):
                    if np.random.random() < 0.8:
                        edges.append([i, j])
        
        current_start += num_clusters * cluster_size
    
    # Add some inter-cluster connections
    for _ in range(num_nodes // 5):
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j:
            edges.append([i, j])
    
    # Convert to TGraph
    edges = list(set(tuple(sorted(edge)) for edge in edges))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    return TGraph(edge_index, num_nodes=num_nodes)


def demo_basic_hierarchical_embedding():
    """Demonstrate basic hierarchical embedding"""
    print("=" * 60)
    print("BASIC HIERARCHICAL EMBEDDING DEMO")
    print("=" * 60)
    
    # Create hierarchical graph
    graph = create_hierarchical_graph(num_nodes=2000, num_levels=3)
    print(f"Created hierarchical graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Create hierarchical embedder
    embedder = create_hierarchical_embedder(
        embedding_method="spectral",
        embed_dim=32,
        max_patch_size=300,
        max_num_patches=10,
        alignment_method="l2g",
        verbose=True
    )
    
    # Compute embedding
    print("\nComputing hierarchical embedding...")
    start_time = time.time()
    embedding = embedder.embed_graph(graph)
    end_time = time.time()
    
    print(f"\nResults:")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Computation time: {end_time - start_time:.2f}s")
    print(f"  Statistics: {embedder.get_statistics()}")
    
    return embedding, embedder


def demo_embedding_methods_comparison():
    """Compare different embedding methods"""
    print("\n" + "=" * 60)
    print("EMBEDDING METHODS COMPARISON")
    print("=" * 60)
    
    graph = create_hierarchical_graph(num_nodes=1500, num_levels=2)
    
    methods = {
        "spectral": SpectralEmbedding(embed_dim=32),
        "random_walk": RandomWalkEmbedding(embed_dim=32, walk_length=10)
    }
    
    results = {}
    
    for method_name, embedding_method in methods.items():
        print(f"\n--- Testing {method_name} embedding ---")
        
        try:
            embedder = HierarchicalEmbedder(
                embedding_method=embedding_method,
                max_patch_size=400,
                max_num_patches=8,
                verbose=False
            )
            
            start_time = time.time()
            embedding = embedder.embed_graph(graph)
            end_time = time.time()
            
            stats = embedder.get_statistics()
            
            results[method_name] = {
                'embedding': embedding,
                'time': end_time - start_time,
                'stats': stats
            }
            
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Time: {end_time - start_time:.2f}s")
            print(f"  Patches created: {stats['total_patches_created']}")
            print(f"  Max recursion depth: {stats['max_recursion_depth']}")
            print(f"  Embedding computations: {stats['embedding_computations']}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[method_name] = {'error': str(e)}
    
    return results


def demo_parameter_scaling():
    """Demonstrate how parameters affect computational complexity"""
    print("\n" + "=" * 60)
    print("PARAMETER SCALING DEMO")
    print("=" * 60)
    
    graph = create_hierarchical_graph(num_nodes=2500, num_levels=3)
    
    # Test different max_patch_size values
    patch_sizes = [200, 500, 1000, 2000]
    
    print(f"Testing on graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    print("\nmax_patch_size | time | patches | depth | embedding_ops | alignment_ops")
    print("-" * 75)
    
    for max_size in patch_sizes:
        try:
            embedder = create_hierarchical_embedder(
                embedding_method="spectral",
                embed_dim=24,
                max_patch_size=max_size,
                verbose=False
            )
            
            start_time = time.time()
            embedding = embedder.embed_graph(graph)
            end_time = time.time()
            
            stats = embedder.get_statistics()
            
            print(f"{max_size:13d} | {end_time - start_time:4.1f}s | "
                  f"{stats['total_patches_created']:7d} | "
                  f"{stats['max_recursion_depth']:5d} | "
                  f"{stats['embedding_computations']:13d} | "
                  f"{stats['alignment_computations']:13d}")
            
        except Exception as e:
            print(f"{max_size:13d} | ERROR: {e}")


def demo_adaptive_vs_fixed():
    """Compare adaptive vs fixed parameter hierarchical embedding"""
    print("\n" + "=" * 60)
    print("ADAPTIVE VS FIXED PARAMETERS DEMO")
    print("=" * 60)
    
    # Create graphs with different characteristics
    graphs = {
        "sparse": create_hierarchical_graph(num_nodes=1800, num_levels=4),
        "dense": create_hierarchical_graph(num_nodes=1200, num_levels=2)
    }
    
    for graph_type, graph in graphs.items():
        print(f"\n--- {graph_type.upper()} GRAPH ---")
        print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
        density = graph.num_edges / (graph.num_nodes * (graph.num_nodes - 1) / 2)
        print(f"Density: {density:.6f}")
        
        # Fixed parameters
        print("\nFixed parameters:")
        fixed_embedder = create_hierarchical_embedder(
            embedding_method="spectral",
            embed_dim=28,
            max_patch_size=400,
            clustering_method="fennel",
            adaptive=False,
            verbose=False
        )
        
        start_time = time.time()
        fixed_embedding = fixed_embedder.embed_graph(graph)
        fixed_time = time.time() - start_time
        fixed_stats = fixed_embedder.get_statistics()
        
        print(f"  Time: {fixed_time:.2f}s")
        print(f"  Patches: {fixed_stats['total_patches_created']}")
        print(f"  Depth: {fixed_stats['max_recursion_depth']}")
        
        # Adaptive parameters
        print("\nAdaptive parameters:")
        adaptive_embedder = create_hierarchical_embedder(
            embedding_method="spectral",
            embed_dim=28,
            max_patch_size=400,
            adaptive=True,
            verbose=False
        )
        
        start_time = time.time()
        adaptive_embedding = adaptive_embedder.embed_graph(graph)
        adaptive_time = time.time() - start_time
        adaptive_stats = adaptive_embedder.get_statistics()
        
        print(f"  Time: {adaptive_time:.2f}s")
        print(f"  Patches: {adaptive_stats['total_patches_created']}")
        print(f"  Depth: {adaptive_stats['max_recursion_depth']}")
        print(f"  Speedup: {fixed_time / adaptive_time:.2f}x")


def demo_scalability():
    """Demonstrate scalability to large graphs"""
    print("\n" + "=" * 60)
    print("SCALABILITY DEMO")
    print("=" * 60)
    
    # Test different graph sizes
    sizes = [1000, 2000, 4000, 8000]
    
    print("Graph Size | Time | Memory Peak | Patches | Depth | Embed Ops")
    print("-" * 65)
    
    for size in sizes:
        try:
            # Create graph
            graph = create_hierarchical_graph(num_nodes=size, num_levels=3)
            
            # Create embedder with size-appropriate parameters
            max_patch_size = min(500, size // 4)
            
            embedder = create_hierarchical_embedder(
                embedding_method="spectral",
                embed_dim=32,
                max_patch_size=max_patch_size,
                adaptive=True,
                verbose=False
            )
            
            # Measure memory usage (rough estimate)
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            embedding = embedder.embed_graph(graph)
            end_time = time.time()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_peak = memory_after - memory_before
            
            stats = embedder.get_statistics()
            
            print(f"{size:10d} | {end_time - start_time:4.1f}s | "
                  f"{memory_peak:10.1f}MB | "
                  f"{stats['total_patches_created']:7d} | "
                  f"{stats['max_recursion_depth']:5d} | "
                  f"{stats['embedding_computations']:9d}")
            
        except Exception as e:
            print(f"{size:10d} | ERROR: {e}")


def visualize_hierarchical_structure(embedder, title="Hierarchical Embedding Structure"):
    """Visualize the hierarchical structure created by the embedder"""
    stats = embedder.get_statistics()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Statistics overview
    stat_names = ['Patches Created', 'Max Depth', 'Embedding Ops', 'Alignment Ops']
    stat_values = [
        stats['total_patches_created'],
        stats['max_recursion_depth'],
        stats['embedding_computations'],
        stats['alignment_computations']
    ]
    
    ax1.bar(stat_names, stat_values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    ax1.set_title('Computation Statistics')
    ax1.set_ylabel('Count')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Theoretical complexity comparison
    sizes = np.logspace(2, 4, 20)  # 100 to 10,000 nodes
    
    # Direct embedding: O(n^3) for spectral
    direct_complexity = sizes ** 3
    
    # Hierarchical: roughly O(n log n) for decomposition + O(k^3) for small patches
    hierarchical_complexity = sizes * np.log(sizes) + (sizes / 500) * (500 ** 3)
    
    ax2.loglog(sizes, direct_complexity, 'r-', label='Direct Embedding O(n³)', linewidth=2)
    ax2.loglog(sizes, hierarchical_complexity, 'b-', label='Hierarchical O(n log n + k³)', linewidth=2)
    ax2.set_xlabel('Graph Size (nodes)')
    ax2.set_ylabel('Computational Complexity')
    ax2.set_title('Complexity Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage pattern
    ax3.text(0.5, 0.5, f"Max Recursion Depth: {stats['max_recursion_depth']}\n"
                       f"Total Patches: {stats['total_patches_created']}\n"
                       f"Embedding Computations: {stats['embedding_computations']}\n"
                       f"Alignment Steps: {stats['alignment_computations']}\n"
                       f"Total Time: {stats['total_time']:.2f}s",
             transform=ax3.transAxes, fontsize=12, verticalalignment='center',
             horizontalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax3.set_title('Detailed Statistics')
    ax3.axis('off')
    
    # Plot 4: Hierarchical decomposition tree (conceptual)
    # This is a simplified visualization of the recursive structure
    levels = stats['max_recursion_depth'] + 1
    x_positions = []
    y_positions = []
    
    for level in range(levels):
        num_nodes_at_level = 2 ** level
        y = levels - level - 1
        for node in range(num_nodes_at_level):
            x = node * (10 / max(1, num_nodes_at_level - 1)) if num_nodes_at_level > 1 else 5
            x_positions.append(x)
            y_positions.append(y)
    
    ax4.scatter(x_positions, y_positions, s=100, alpha=0.7, c=y_positions, cmap='viridis')
    
    # Add connections between levels
    for level in range(levels - 1):
        num_nodes_current = 2 ** level
        num_nodes_next = 2 ** (level + 1)
        y_current = levels - level - 1
        y_next = levels - level - 2
        
        for i in range(num_nodes_current):
            x_current = i * (10 / max(1, num_nodes_current - 1)) if num_nodes_current > 1 else 5
            
            # Connect to children
            for j in range(2):  # Each node has 2 children
                child_idx = i * 2 + j
                if child_idx < num_nodes_next:
                    x_child = child_idx * (10 / max(1, num_nodes_next - 1)) if num_nodes_next > 1 else 5
                    ax4.plot([x_current, x_child], [y_current, y_next], 'k-', alpha=0.3)
    
    ax4.set_title('Hierarchical Decomposition Tree')
    ax4.set_xlabel('Spatial Position')
    ax4.set_ylabel('Recursion Level')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig


def main():
    """Run all demonstrations"""
    print("Hierarchical Graph Embedding Demonstration")
    print("This demo shows the hierarchical embedding approach for large graphs")
    
    try:
        # Basic demo
        embedding, embedder = demo_basic_hierarchical_embedding()
        
        # Method comparison
        demo_embedding_methods_comparison()
        
        # Parameter scaling
        demo_parameter_scaling()
        
        # Adaptive vs fixed
        demo_adaptive_vs_fixed()
        
        # Scalability test
        try:
            demo_scalability()
        except ImportError:
            print("Scalability demo skipped (psutil not available)")
        
        # Visualization
        try:
            fig = visualize_hierarchical_structure(embedder, "Hierarchical Embedding Analysis")
            
            # Save plot
            output_path = Path(__file__).parent / "hierarchical_embedding_analysis.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to: {output_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Visualization skipped: {e}")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("Key advantages of hierarchical embedding:")
        print("1. Scalable to large graphs through recursive decomposition")
        print("2. Balances embedding complexity with decomposition complexity")
        print("3. Adaptive parameter selection for different graph types")
        print("4. Maintains embedding quality through hierarchical alignment")
        print("5. Memory efficient through patch-based processing")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()