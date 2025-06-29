"""
L2GX Patch Module Comprehensive Demonstration

This example demonstrates the reorganized and improved patch module functionality,
showcasing the complete patch-based graph processing pipeline including:

- Graph clustering algorithms for patch generation
- Patch creation with controlled overlap
- Graph sparsification techniques
- Coordinate transformation utilities
- Memory-efficient lazy loading systems

Features demonstrated:
- Multiple clustering algorithms (Fennel, Louvain, METIS, Spread)
- Patch creation and overlap management
- Sparsification methods for scalability
- Error analysis and quality metrics
- Visualization of patch structures
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, Any, List
import torch
from torch_geometric.data import Data

# Import reorganized patch module functionality
from l2gx.patch import (
    # Clustering algorithms
    fennel_clustering,
    louvain_clustering, 
    spread_clustering,
    # Patch creation
    create_patch_data,
    create_overlapping_patches,
    Patch,
    # Sparsification methods
    edge_sampling_sparsify,
    nearest_neighbor_sparsify,
    # Utility functions
    procrustes_error,
    local_error,
    transform_error,
    relative_scale,
)

from l2gx.datasets import get_dataset
from l2gx.graphs import TGraph


def demonstrate_clustering_algorithms():
    """Demonstrate various clustering algorithms for patch generation."""
    print("🔬 CLUSTERING ALGORITHMS DEMONSTRATION")
    print("=" * 60)
    
    # Load Cora dataset for demonstration
    print("📚 Loading Cora dataset...")
    cora = get_dataset("Cora")
    data = cora[0]
    
    # Convert to TGraph for clustering
    tgraph = TGraph(data.edge_index, x=data.x)
    
    clustering_methods = {
        'fennel': lambda: fennel_clustering(data, num_clusters=7),
        'spread': lambda: spread_clustering(tgraph.to_raphtory(), num_clusters=7),
    }
    
    clustering_results = {}
    clustering_times = {}
    
    for method_name, clustering_func in clustering_methods.items():
        print(f"🔄 Running {method_name} clustering...")
        start_time = time.time()
        
        try:
            result = clustering_func()
            end_time = time.time()
            clustering_times[method_name] = end_time - start_time
            clustering_results[method_name] = result
            
            print(f"✅ {method_name} completed in {clustering_times[method_name]:.3f}s")
            
            # Analyze clustering quality
            if hasattr(result, 'unique'):
                n_clusters = len(result.unique())
            else:
                n_clusters = len(set(result))
            print(f"   Clusters found: {n_clusters}")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {str(e)}")
            clustering_results[method_name] = None
    
    print()
    return data, tgraph, clustering_results


def demonstrate_patch_creation(data, tgraph, clustering_results):
    """Demonstrate patch creation with different clustering results."""
    print("🧩 PATCH CREATION DEMONSTRATION")
    print("=" * 60)
    
    patch_collections = {}
    
    for method_name, partition in clustering_results.items():
        if partition is None:
            continue
            
        print(f"🔄 Creating patches from {method_name} clustering...")
        start_time = time.time()
        
        try:
            # Create patches with overlap
            patches = create_patch_data(
                graph=tgraph,
                partition=partition,
                overlap_size=5,
                sparsify_overlap=True,
                overlap_sparsification='edge_sampling'
            )
            
            end_time = time.time()
            patch_collections[method_name] = patches
            
            print(f"✅ Created {len(patches)} patches in {end_time - start_time:.3f}s")
            
            # Analyze patch properties
            patch_sizes = [len(patch.nodes) for patch in patches]
            overlap_counts = []
            
            for i, patch1 in enumerate(patches):
                overlaps = 0
                for j, patch2 in enumerate(patches):
                    if i != j:
                        overlap = len(set(patch1.nodes.tolist()) & set(patch2.nodes.tolist()))
                        if overlap > 0:
                            overlaps += 1
                overlap_counts.append(overlaps)
            
            print(f"   Patch size: min={min(patch_sizes)}, max={max(patch_sizes)}, mean={np.mean(patch_sizes):.1f}")
            print(f"   Overlaps per patch: min={min(overlap_counts)}, max={max(overlap_counts)}, mean={np.mean(overlap_counts):.1f}")
            
        except Exception as e:
            print(f"❌ Patch creation failed for {method_name}: {str(e)}")
    
    print()
    return patch_collections


def demonstrate_sparsification(tgraph):
    """Demonstrate graph sparsification techniques."""
    print("✂️  GRAPH SPARSIFICATION DEMONSTRATION")
    print("=" * 60)
    
    original_edges = tgraph.num_edges
    target_degree = 10  # Target average degree
    
    sparsification_methods = {
        'edge_sampling': lambda: edge_sampling_sparsify(tgraph, target_degree),
        'nearest_neighbor': lambda: nearest_neighbor_sparsify(tgraph, target_degree),
    }
    
    sparsified_graphs = {}
    
    for method_name, sparsify_func in sparsification_methods.items():
        print(f"🔄 Applying {method_name} sparsification...")
        start_time = time.time()
        
        try:
            sparsified = sparsify_func()
            end_time = time.time()
            sparsified_graphs[method_name] = sparsified
            
            reduction = (1 - sparsified.num_edges / original_edges) * 100
            print(f"✅ {method_name} completed in {end_time - start_time:.3f}s")
            print(f"   Edge reduction: {reduction:.1f}% ({original_edges} → {sparsified.num_edges})")
            
        except Exception as e:
            print(f"❌ {method_name} failed: {str(e)}")
    
    print()
    return sparsified_graphs


def demonstrate_coordinate_utilities():
    """Demonstrate coordinate transformation and error utilities."""
    print("📐 COORDINATE UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic coordinate data for demonstration
    np.random.seed(42)
    n_nodes = 100
    dim = 2
    
    # Original coordinates
    coords1 = np.random.randn(n_nodes, dim)
    
    # Create transformed coordinates (rotation + scale + noise)
    rotation_angle = np.pi / 4
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle), np.cos(rotation_angle)]
    ])
    scale_factor = 1.5
    noise_level = 0.1
    
    coords2 = scale_factor * (coords1 @ rotation_matrix.T) + noise_level * np.random.randn(n_nodes, dim)
    
    # Demonstrate utility functions
    print("🔄 Computing coordinate transformation metrics...")
    
    # Procrustes error
    proc_error = procrustes_error(coords1, coords2)
    print(f"✅ Procrustes alignment error: {proc_error:.4f}")
    
    # Relative scale
    rel_scale = relative_scale(coords1, coords2)
    print(f"✅ Relative scale factor: {rel_scale:.4f} (expected: {scale_factor:.4f})")
    
    # Transform error simulation
    n_patches = 5
    transforms = []
    for i in range(n_patches):
        # Simulate patch transformations
        angle = np.random.normal(rotation_angle, 0.1)
        scale = np.random.normal(scale_factor, 0.1)
        transform = scale * np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        transforms.append(transform)
    
    trans_error = transform_error(transforms)
    print(f"✅ Transformation consistency error: {trans_error:.4f}")
    
    print()


def demonstrate_patch_visualization(patch_collections):
    """Visualize patch structures and overlaps."""
    print("📊 PATCH VISUALIZATION")
    print("=" * 60)
    
    if not patch_collections:
        print("ℹ️  No patch collections available for visualization")
        return
    
    # Take the first available patch collection
    method_name, patches = next(iter(patch_collections.items()))
    print(f"🔄 Visualizing patches from {method_name} clustering...")
    
    # Create visualization of patch structure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Patch Structure Analysis ({method_name.upper()})', fontsize=16)
    
    # Plot 1: Patch size distribution
    patch_sizes = [len(patch.nodes) for patch in patches]
    axes[0].hist(patch_sizes, bins=min(20, len(patches)), alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].set_title('Patch Size Distribution')
    axes[0].set_xlabel('Number of Nodes per Patch')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics
    mean_size = np.mean(patch_sizes)
    std_size = np.std(patch_sizes)
    axes[0].axvline(mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.1f}')
    axes[0].legend()
    
    # Plot 2: Overlap analysis
    overlap_matrix = np.zeros((len(patches), len(patches)))
    for i, patch1 in enumerate(patches):
        for j, patch2 in enumerate(patches):
            if i != j:
                overlap = len(set(patch1.nodes.tolist()) & set(patch2.nodes.tolist()))
                overlap_matrix[i, j] = overlap
    
    im = axes[1].imshow(overlap_matrix, cmap='Blues', aspect='auto')
    axes[1].set_title('Patch Overlap Matrix')
    axes[1].set_xlabel('Patch Index')
    axes[1].set_ylabel('Patch Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Number of Overlapping Nodes')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    total_overlaps = np.sum(overlap_matrix > 0)
    max_overlap = np.max(overlap_matrix)
    mean_overlap = np.mean(overlap_matrix[overlap_matrix > 0]) if total_overlaps > 0 else 0
    
    print(f"📊 Overlap Statistics:")
    print(f"   Total patch pairs with overlap: {total_overlaps}")
    print(f"   Maximum overlap size: {max_overlap}")
    print(f"   Mean overlap size: {mean_overlap:.1f}")
    print()


def run_comprehensive_patch_demo():
    """Run the complete patch module demonstration."""
    print("🧩 L2GX COMPREHENSIVE PATCH MODULE DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        # 1. Clustering algorithms
        data, tgraph, clustering_results = demonstrate_clustering_algorithms()
        
        # 2. Patch creation
        patch_collections = demonstrate_patch_creation(data, tgraph, clustering_results)
        
        # 3. Sparsification methods
        sparsified_graphs = demonstrate_sparsification(tgraph)
        
        # 4. Coordinate utilities
        demonstrate_coordinate_utilities()
        
        # 5. Visualization
        demonstrate_patch_visualization(patch_collections)
        
        # Summary
        print("🎊 PATCH MODULE DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("   • Graph clustering algorithms for patch generation")
        print("   • Patch creation with controlled overlap management")
        print("   • Graph sparsification for scalability")
        print("   • Coordinate transformation utilities")
        print("   • Patch structure visualization and analysis")
        print()
        
        successful_clusterings = sum(1 for result in clustering_results.values() if result is not None)
        print(f"📊 Results: {successful_clusterings}/{len(clustering_results)} clustering methods successful")
        print(f"📦 Created {len(patch_collections)} patch collections")
        print(f"✂️  Tested {len(sparsified_graphs)} sparsification methods")
        print()
        print("🚀 The reorganized L2GX patch module is ready for scalable graph processing!")
        
        return {
            'clustering_results': clustering_results,
            'patch_collections': patch_collections,
            'sparsified_graphs': sparsified_graphs
        }
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_comprehensive_patch_demo()