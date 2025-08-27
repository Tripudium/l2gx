#!/usr/bin/env python3
"""
L2GX Embedding Examples

This script demonstrates the three main embedding approaches in L2GX:
1. Simple Embedding: Direct embedding of the full graph
2. Patched Embedding: Decompose into patches and align globally
3. Hierarchical Embedding: Tree structure with bottom-up alignment

Run this script to see all embedding methods in action.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for L2GX imports if running directly
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

import numpy as np
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner
from l2gx.graphs import TGraph


def load_example_data():
    """Load the Cora dataset for examples."""
    print("Loading Cora dataset...")
    dataset = get_dataset("Cora")
    pg_data = dataset.to("torch-geometric")
    data = TGraph.from_tg(pg_data)
    print(f"Loaded {data.num_nodes} nodes, {data.num_edges} edges, {data.y.max().item() + 1} classes")
    return data


def simple_embedding_example(data):
    """Example 1: Simple embedding of the full graph."""
    print("\n" + "="*60)
    print("SIMPLE EMBEDDING EXAMPLE")
    print("="*60)
    
    print("Using VGAE to embed the full graph...")
    start_time = time.time()
    
    # Create embedder for full graph
    embedder = get_embedding(
        "vgae",                    # Method: vgae, gae, svd, dgi, graphsage
        embedding_dim=64,          # Smaller for faster execution
        epochs=50,                 # Reduced for demo
        learning_rate=0.001,
        patience=10
    )
    
    # Compute embedding
    embedding = embedder.fit_transform(data.to_tg())
    elapsed_time = time.time() - start_time
    
    print(f"✅ Simple embedding completed!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")
    print(f"   Time: {elapsed_time:.2f} seconds")
    
    return embedding


def patched_l2g_example(data):
    """Example 2: Patched embedding with L2G alignment."""
    print("\n" + "="*60)
    print("PATCHED EMBEDDING WITH L2G ALIGNMENT")
    print("="*60)
    
    print("Creating L2G aligner with randomized sketching...")
    # Create and configure L2G aligner
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "randomized"  # Enable randomization
    l2g_aligner.sketch_method = "rademacher"      # Sketch method
    
    print("Creating patched embedder...")
    # Create patched embedder
    embedder = get_embedding(
        "patched",
        embedding_dim=64,           # Smaller for faster execution
        aligner=l2g_aligner,        # Required aligner object
        num_patches=8,              # Number of patches to create
        base_method="svd",          # Fast method for demo
        clustering_method="metis",  # Graph partitioning method
        min_overlap=10,             # Minimum overlap between patches
        target_overlap=20,          # Target overlap size
        verbose=False               # Set to True to see patch creation details
    )
    
    start_time = time.time()
    embedding = embedder.fit_transform(data.to_tg())
    elapsed_time = time.time() - start_time
    
    # Get patch information
    patch_info = embedder.get_patch_info()
    
    print(f"✅ Patched L2G embedding completed!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")
    print(f"   Number of patches: {patch_info['num_patches']}")
    print(f"   Mean patch size: {patch_info['patch_sizes']['mean']:.1f}")
    print(f"   Time: {elapsed_time:.2f} seconds")
    
    return embedding


def patched_geo_example(data):
    """Example 3: Patched embedding with Geo alignment."""
    print("\n" + "="*60)
    print("PATCHED EMBEDDING WITH GEO ALIGNMENT")
    print("="*60)
    
    print("Creating Geo aligner with orthogonal manifold optimization...")
    # Create and configure Geo aligner
    geo_aligner = get_aligner("geo", 
        method="orthogonal",        # orthogonal or euclidean
        use_scale=True,             # Enable scale optimization
        num_epochs=5,               # Optimization epochs (reduced for demo)
        learning_rate=0.01,         # Learning rate
        verbose=False               # Set to True to see optimization progress
    )
    
    print("Creating patched embedder with Geo alignment...")
    # Create patched embedder with Geo alignment
    embedder = get_embedding(
        "patched",
        embedding_dim=64,           # Smaller for faster execution
        aligner=geo_aligner,        # Custom configured aligner
        num_patches=5,              # Fewer patches work better with Geo
        base_method="svd",          # Fast base method
        min_overlap=15,             # Higher overlap for better alignment
        target_overlap=25,
        verbose=False
    )
    
    start_time = time.time()
    embedding = embedder.fit_transform(data.to_tg())
    elapsed_time = time.time() - start_time
    
    patch_info = embedder.get_patch_info()
    
    print(f"✅ Patched Geo embedding completed!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")
    print(f"   Number of patches: {patch_info['num_patches']}")
    print(f"   Time: {elapsed_time:.2f} seconds")
    
    return embedding


def hierarchical_example(data):
    """Example 4: Hierarchical embedding."""
    print("\n" + "="*60)
    print("HIERARCHICAL EMBEDDING")
    print("="*60)
    
    print("Creating hierarchical embedder...")
    # Create aligner (currently hierarchical uses Procrustes regardless)
    l2g_aligner = get_aligner("l2g")
    
    # Create hierarchical embedder
    embedder = get_embedding(
        "hierarchical",
        embedding_dim=64,           # Smaller for faster execution
        aligner=l2g_aligner,        # Required (but Procrustes is used)
        max_patch_size=200,         # Smaller patches for faster execution
        branching_factor=3,         # Children per internal node (ternary tree)
        base_method="svd",          # Fast method for leaves
        clustering_method="metis",  # Graph partitioning method
        max_levels=3,               # Limit tree depth for demo
        min_overlap=32,             # Overlap between patches
        target_overlap=64,
        verbose=False               # Set to True to see tree construction
    )
    
    start_time = time.time()
    embedding = embedder.fit_transform(data.to_tg())
    elapsed_time = time.time() - start_time
    
    # Get tree structure information
    tree_info = embedder.get_tree_structure()
    
    print(f"✅ Hierarchical embedding completed!")
    print(f"   Shape: {embedding.shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(embedding, axis=1)):.3f}")
    print(f"   Tree depth: {tree_info['max_depth']}")
    print(f"   Number of leaves: {tree_info['num_leaves']}")
    print(f"   Time: {elapsed_time:.2f} seconds")
    print("   Note: Currently uses Procrustes alignment regardless of aligner")
    
    return embedding


def compare_embeddings(embeddings):
    """Compare the different embedding results."""
    print("\n" + "="*60)
    print("EMBEDDING COMPARISON")
    print("="*60)
    
    print(f"{'Method':<20} {'Shape':<12} {'Mean Norm':<12} {'Std Norm':<12}")
    print("-" * 60)
    
    for method, embedding in embeddings.items():
        mean_norm = np.mean(np.linalg.norm(embedding, axis=1))
        std_norm = np.std(np.linalg.norm(embedding, axis=1))
        print(f"{method:<20} {str(embedding.shape):<12} {mean_norm:<12.3f} {std_norm:<12.3f}")


def main():
    """Run all embedding examples."""
    print("L2GX Embedding Examples")
    print("=======================")
    print()
    print("This script demonstrates three main embedding approaches:")
    print("1. Simple Embedding: Direct VGAE embedding")
    print("2. Patched L2G: Patches with L2G alignment")
    print("3. Patched Geo: Patches with Geo alignment")
    print("4. Hierarchical: Tree structure with Procrustes alignment")
    
    # Load data
    data = load_example_data()
    
    # Store results for comparison
    embeddings = {}
    
    # Run examples
    try:
        embeddings['Simple VGAE'] = simple_embedding_example(data)
    except Exception as e:
        print(f"❌ Simple embedding failed: {e}")
    
    try:
        embeddings['Patched L2G'] = patched_l2g_example(data)
    except Exception as e:
        print(f"❌ Patched L2G embedding failed: {e}")
    
    try:
        embeddings['Patched Geo'] = patched_geo_example(data)
    except Exception as e:
        print(f"❌ Patched Geo embedding failed: {e}")
    
    try:
        embeddings['Hierarchical'] = hierarchical_example(data)
    except Exception as e:
        print(f"❌ Hierarchical embedding failed: {e}")
    
    # Compare results
    if embeddings:
        compare_embeddings(embeddings)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Successfully computed {len(embeddings)} embeddings!")
        print()
        print("Next steps:")
        print("- Examine the embedding_guide.rst documentation for more details")
        print("- Try different parameters and alignment methods")
        print("- Use configuration files for reproducible experiments")
        print("- Apply embeddings to downstream tasks like classification")
    else:
        print("\n❌ No embeddings completed successfully. Please check the error messages above.")


if __name__ == "__main__":
    main()