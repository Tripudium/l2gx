#!/usr/bin/env python3
"""
Simple L2GX Embedding Demo

Basic demonstration of L2GX embedding methods that are reliable and fast.
Perfect for getting started with the L2GX API.
"""

import sys
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


def main():
    """Simple demonstration of embedding methods."""
    print("L2GX Simple Embedding Demo")
    print("==========================")
    print("Demonstrating basic embedding functionality...\n")
    
    # Load dataset
    print("Loading Cora dataset...")
    dataset = get_dataset("Cora")
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    print(f"Loaded {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    results = {}
    
    # 1. Simple SVD embedding
    print("1. Simple SVD Embedding")
    print("-" * 25)
    print("Direct SVD embedding of the full adjacency matrix...")
    embedder = get_embedding("svd", embedding_dim=64)
    results['SVD'] = embedder.fit_transform(data.to_tg())
    print(f"✅ Completed! Shape: {results['SVD'].shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(results['SVD'], axis=1)):.3f}\n")
    
    # 2. Patched embedding with simple configuration
    print("2. Patched Embedding")
    print("-" * 20)
    print("Creating patches and using Geo alignment...")
    
    # Use Geo alignment which is more stable
    geo_aligner = get_aligner("geo", method="orthogonal", use_scale=True)
    
    embedder = get_embedding(
        "patched", 
        embedding_dim=64, 
        aligner=geo_aligner,
        num_patches=4,           # Small number for stability
        base_method="svd",       # Fast and reliable
        min_overlap=20,
        target_overlap=40,
        verbose=False
    )
    
    try:
        results['Patched'] = embedder.fit_transform(data.to_tg())
        patch_info = embedder.get_patch_info()
        print(f"✅ Completed! Shape: {results['Patched'].shape}")
        print(f"   Mean norm: {np.mean(np.linalg.norm(results['Patched'], axis=1)):.3f}")
        print(f"   Created {patch_info['num_patches']} patches, avg size: {patch_info['patch_sizes']['mean']:.1f}\n")
    except Exception as e:
        print(f"❌ Patched embedding failed: {e}\n")
    
    # 3. Hierarchical embedding
    print("3. Hierarchical Embedding")
    print("-" * 25)
    print("Creating tree structure with binary splits...")
    
    # Hierarchical uses Procrustes regardless of aligner
    dummy_aligner = get_aligner("geo")  # Required but not used
    
    embedder = get_embedding(
        "hierarchical",
        embedding_dim=64,
        aligner=dummy_aligner,
        max_patch_size=100,      # Small for fast demo
        branching_factor=2,      # Binary tree
        base_method="svd",
        max_levels=2,            # Limit depth
        verbose=False
    )
    
    try:
        results['Hierarchical'] = embedder.fit_transform(data.to_tg())
        tree_info = embedder.get_tree_structure()
        print(f"✅ Completed! Shape: {results['Hierarchical'].shape}")
        print(f"   Mean norm: {np.mean(np.linalg.norm(results['Hierarchical'], axis=1)):.3f}")
        print(f"   Tree depth: {tree_info['max_depth']}, leaves: {tree_info['num_leaves']}")
        print("   Note: Uses Procrustes alignment regardless of aligner\n")
    except Exception as e:
        print(f"❌ Hierarchical embedding failed: {e}\n")
    
    # Summary
    print("Summary")
    print("=======")
    if results:
        print(f"{'Method':<12} {'Shape':<12} {'Mean Norm':<10}")
        print("-" * 35)
        for method, embedding in results.items():
            mean_norm = np.mean(np.linalg.norm(embedding, axis=1))
            print(f"{method:<12} {str(embedding.shape):<12} {mean_norm:<10.3f}")
        
        print(f"\n✅ Successfully computed {len(results)} embeddings!")
    else:
        print("❌ No embeddings completed successfully.")
    
    print("\nNext Steps:")
    print("----------")
    print("1. Read the full documentation: docs/source/embedding_guide.rst")
    print("2. Try configuration-based experiments: scripts/embedding/")
    print("3. Experiment with neural methods: examples/embedding_examples.py")
    print("4. Customize aligners and parameters for your specific needs")


if __name__ == "__main__":
    main()