#!/usr/bin/env python3
"""
Patch Generation Demonstration

This script demonstrates the high-level patch generation functions,
showing different ways to create patches from graphs with various parameters.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add L2G to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from l2gx.graphs import TGraph
from l2gx.patch import (
    generate_patches,
    generate_patches_by_size,
    generate_patches_adaptive,
    estimate_patch_parameters,
    list_clustering_methods
)


def create_example_graph(num_nodes: int = 1000, avg_degree: int = 8, seed: int = 42) -> TGraph:
    """Create an example graph with community structure"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create clusters
    cluster_size = num_nodes // 10
    edges = []
    
    for cluster_start in range(0, num_nodes, cluster_size):
        cluster_end = min(cluster_start + cluster_size, num_nodes)
        
        # Dense connections within cluster
        for i in range(cluster_start, cluster_end):
            for j in range(i + 1, min(i + avg_degree // 2, cluster_end)):
                if np.random.random() < 0.8:  # High intra-cluster connectivity
                    edges.append([i, j])
    
    # Sparse inter-cluster connections
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


def demo_basic_patch_generation():
    """Demonstrate basic patch generation with different parameters"""
    print("=" * 60)
    print("BASIC PATCH GENERATION DEMO")
    print("=" * 60)
    
    # Create example graph
    graph = create_example_graph(num_nodes=2000, avg_degree=10)
    print(f"Created graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Method 1: Generate patches by target size
    print("\n1. Generate patches by target size:")
    patches, patch_graph = generate_patches(
        graph,
        patch_size=100,
        clustering_method="fennel",
        sparsify_method="resistance"
    )
    
    patch_sizes = [len(patch.nodes) for patch in patches]
    print(f"   Result: {len(patches)} patches")
    print(f"   Patch sizes: [{min(patch_sizes)}, {max(patch_sizes)}], avg: {np.mean(patch_sizes):.1f}")
    
    # Method 2: Generate patches by number
    print("\n2. Generate patches by target number:")
    patches, patch_graph = generate_patches(
        graph,
        num_patches=25,
        clustering_method="louvain",
        sparsify_method="none"
    )
    
    patch_sizes = [len(patch.nodes) for patch in patches]
    print(f"   Result: {len(patches)} patches")
    print(f"   Patch sizes: [{min(patch_sizes)}, {max(patch_sizes)}], avg: {np.mean(patch_sizes):.1f}")


def demo_different_clustering_methods():
    """Compare different clustering methods"""
    print("\n" + "=" * 60)
    print("CLUSTERING METHODS COMPARISON")
    print("=" * 60)
    
    graph = create_example_graph(num_nodes=1500, avg_degree=8)
    
    # List available methods
    methods = list_clustering_methods()
    print("Available clustering methods:")
    for method, description in methods.items():
        print(f"  {method}: {description}")
    
    print(f"\nTesting on graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Test different methods
    test_methods = ["fennel", "louvain", "metis", "spread"]
    
    for method in test_methods:
        print(f"\n--- Testing {method} ---")
        try:
            patches, _ = generate_patches(
                graph,
                patch_size=120,
                clustering_method=method,
                verbose=False
            )
            
            patch_sizes = [len(patch.nodes) for patch in patches]
            print(f"  Patches: {len(patches)}")
            print(f"  Size range: [{min(patch_sizes)}, {max(patch_sizes)}]")
            print(f"  Average size: {np.mean(patch_sizes):.1f}")
            print(f"  Size std: {np.std(patch_sizes):.1f}")
            
        except Exception as e:
            print(f"  Failed: {e}")


def demo_adaptive_patch_generation():
    """Demonstrate adaptive patch generation"""
    print("\n" + "=" * 60)
    print("ADAPTIVE PATCH GENERATION DEMO")
    print("=" * 60)
    
    graph = create_example_graph(num_nodes=3000, avg_degree=12)
    print(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    # Method 1: Size-controlled generation
    print("\n1. Size-controlled patch generation:")
    patches, _ = generate_patches_by_size(
        graph,
        target_patch_size=150,
        size_tolerance=0.15,  # ±15%
        clustering_method="fennel"
    )
    
    patch_sizes = [len(patch.nodes) for patch in patches]
    print(f"   Target: 150 nodes per patch")
    print(f"   Result: {len(patches)} patches, avg size: {np.mean(patch_sizes):.1f}")
    
    # Method 2: Fully adaptive generation
    print("\n2. Fully adaptive patch generation:")
    patches, _ = generate_patches_adaptive(
        graph,
        max_patch_size=200,
        min_patch_size=50,
        clustering_method="louvain"
    )
    
    patch_sizes = [len(patch.nodes) for patch in patches]
    print(f"   Size constraints: [50, 200] nodes")
    print(f"   Result: {len(patches)} patches, avg size: {np.mean(patch_sizes):.1f}")


def demo_parameter_estimation():
    """Demonstrate parameter estimation utilities"""
    print("\n" + "=" * 60)
    print("PARAMETER ESTIMATION DEMO")
    print("=" * 60)
    
    # Test with different graph sizes
    test_sizes = [500, 2000, 10000]
    
    for num_nodes in test_sizes:
        graph = create_example_graph(num_nodes=num_nodes)
        
        print(f"\nGraph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        
        # Estimate parameters for target patch size
        params = estimate_patch_parameters(graph, target_patch_size=100)
        print(f"  Target patch size 100:")
        print(f"    Recommended patches: {params['num_patches']}")
        print(f"    Estimated size: {params['patch_size']}")
        print(f"    Overlap: min={params['min_overlap']}, target={params['target_overlap']}")
        print(f"    Clustering method: {params['clustering_method']}")
        
        # Estimate parameters for target number of patches
        params = estimate_patch_parameters(graph, target_num_patches=20)
        print(f"  Target 20 patches:")
        print(f"    Estimated patch size: {params['patch_size']}")
        print(f"    Clustering method: {params['clustering_method']}")


def demo_sparsification_methods():
    """Demonstrate different sparsification methods"""
    print("\n" + "=" * 60)
    print("SPARSIFICATION METHODS DEMO")
    print("=" * 60)
    
    graph = create_example_graph(num_nodes=1000, avg_degree=15)
    print(f"Original graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    
    sparsify_methods = ["none", "resistance", "edge_sampling", "knn"]
    
    for method in sparsify_methods:
        print(f"\n--- Sparsification: {method} ---")
        try:
            patches, patch_graph = generate_patches(
                graph,
                patch_size=80,
                clustering_method="fennel",
                sparsify_method=method,
                verbose=False
            )
            
            # Calculate statistics
            total_edges_in_patches = sum(
                len(patch.edge_index[0]) if hasattr(patch, 'edge_index') else 0 
                for patch in patches
            )
            
            print(f"  Patches created: {len(patches)}")
            print(f"  Patch graph edges: {len(patch_graph.nonzero()[0])}")
            
        except Exception as e:
            print(f"  Failed: {e}")


def visualize_patch_sizes(patches, title="Patch Size Distribution"):
    """Create a histogram of patch sizes"""
    patch_sizes = [len(patch.nodes) for patch in patches]
    
    plt.figure(figsize=(10, 6))
    plt.hist(patch_sizes, bins=min(20, len(patches)//2), alpha=0.7, edgecolor='black')
    plt.xlabel('Patch Size (number of nodes)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.axvline(np.mean(patch_sizes), color='red', linestyle='--', 
                label=f'Mean: {np.mean(patch_sizes):.1f}')
    plt.axvline(np.median(patch_sizes), color='green', linestyle='--',
                label=f'Median: {np.median(patch_sizes):.1f}')
    plt.legend()
    
    plt.tight_layout()
    return plt.gcf()


def demo_visualization():
    """Demonstrate patch visualization"""
    print("\n" + "=" * 60)
    print("PATCH VISUALIZATION DEMO")
    print("=" * 60)
    
    graph = create_example_graph(num_nodes=800, avg_degree=10)
    
    # Generate patches with different methods
    methods = ["fennel", "louvain"]
    
    fig, axes = plt.subplots(1, len(methods), figsize=(15, 5))
    if len(methods) == 1:
        axes = [axes]
    
    for i, method in enumerate(methods):
        print(f"Generating patches with {method}...")
        patches, _ = generate_patches(
            graph,
            patch_size=60,
            clustering_method=method,
            verbose=False
        )
        
        patch_sizes = [len(patch.nodes) for patch in patches]
        
        axes[i].hist(patch_sizes, bins=15, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel('Patch Size')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{method.capitalize()} Clustering\n({len(patches)} patches)')
        axes[i].grid(True, alpha=0.3)
        
        # Add mean line
        mean_size = np.mean(patch_sizes)
        axes[i].axvline(mean_size, color='red', linestyle='--', 
                       label=f'Mean: {mean_size:.1f}')
        axes[i].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(__file__).parent / "patch_generation_demo.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Run all demonstrations"""
    print("L2G Patch Generation Demonstration")
    print("This demo shows various ways to generate patches from graphs")
    
    try:
        demo_basic_patch_generation()
        demo_different_clustering_methods()
        demo_adaptive_patch_generation()
        demo_parameter_estimation()
        demo_sparsification_methods()
        
        # Ask if user wants visualization
        try:
            demo_visualization()
        except Exception as e:
            print(f"\nVisualization demo skipped: {e}")
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("Key takeaways:")
        print("1. Use generate_patches() for basic patch generation")
        print("2. Use generate_patches_by_size() for size-controlled generation")
        print("3. Use generate_patches_adaptive() for automatic parameter selection")
        print("4. Use estimate_patch_parameters() to get recommended settings")
        print("5. Different clustering methods work better for different graph types")
        print("6. Sparsification can improve computational efficiency")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()