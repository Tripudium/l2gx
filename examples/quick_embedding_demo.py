#!/usr/bin/env python3
"""
Quick L2GX Embedding Demo

Fast demonstration of L2GX embedding methods using SVD for speed.
Perfect for testing and getting familiar with the API.
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
    """Quick demo of all embedding types."""
    print("L2GX Quick Embedding Demo")
    print("=========================")
    print("Using SVD for fast demonstration...\n")
    
    # Load dataset
    print("Loading Cora dataset...")
    dataset = get_dataset("Cora")
    data = TGraph.from_tg(dataset.to("torch-geometric"))
    print(f"Loaded {data.num_nodes} nodes, {data.num_edges} edges\n")
    
    results = {}
    
    # 1. Simple SVD embedding
    print("1. Simple SVD Embedding")
    print("-" * 25)
    embedder = get_embedding("svd", embedding_dim=64)
    results['Simple SVD'] = embedder.fit_transform(data.to_tg())
    print(f"✅ Shape: {results['Simple SVD'].shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(results['Simple SVD'], axis=1)):.3f}\n")
    
    # 2. Patched embedding with L2G
    print("2. Patched Embedding (L2G)")
    print("-" * 27)
    l2g_aligner = get_aligner("l2g")
    l2g_aligner.randomized_method = "randomized"
    
    embedder = get_embedding(
        "patched", embedding_dim=64, aligner=l2g_aligner,
        num_patches=6, base_method="svd", verbose=False
    )
    results['Patched L2G'] = embedder.fit_transform(data.to_tg())
    patch_info = embedder.get_patch_info()
    print(f"✅ Shape: {results['Patched L2G'].shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(results['Patched L2G'], axis=1)):.3f}")
    print(f"   Patches: {patch_info['num_patches']}, avg size: {patch_info['patch_sizes']['mean']:.1f}\n")
    
    # 3. Patched embedding with Geo  
    print("3. Patched Embedding (Geo)")
    print("-" * 27)
    geo_aligner = get_aligner("geo", method="orthogonal", use_scale=True)
    
    embedder = get_embedding(
        "patched", embedding_dim=64, aligner=geo_aligner,
        num_patches=4, base_method="svd", verbose=False
    )
    results['Patched Geo'] = embedder.fit_transform(data.to_tg())
    print(f"✅ Shape: {results['Patched Geo'].shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(results['Patched Geo'], axis=1)):.3f}\n")
    
    # 4. Hierarchical embedding
    print("4. Hierarchical Embedding")
    print("-" * 25)
    hier_aligner = get_aligner("l2g")  # Uses Procrustes regardless
    
    embedder = get_embedding(
        "hierarchical", embedding_dim=64, aligner=hier_aligner,
        max_patch_size=150, branching_factor=2, base_method="svd",
        max_levels=2, verbose=False
    )
    results['Hierarchical'] = embedder.fit_transform(data.to_tg())
    tree_info = embedder.get_tree_structure()
    print(f"✅ Shape: {results['Hierarchical'].shape}")
    print(f"   Mean norm: {np.mean(np.linalg.norm(results['Hierarchical'], axis=1)):.3f}")
    print(f"   Tree depth: {tree_info['max_depth']}, leaves: {tree_info['num_leaves']}\n")
    
    # Summary comparison
    print("Summary Comparison")
    print("==================")
    print(f"{'Method':<15} {'Shape':<12} {'Mean Norm':<10}")
    print("-" * 40)
    for method, embedding in results.items():
        mean_norm = np.mean(np.linalg.norm(embedding, axis=1))
        print(f"{method:<15} {str(embedding.shape):<12} {mean_norm:<10.3f}")
    
    print(f"\n✅ All {len(results)} embedding methods completed successfully!")
    print("\nNext steps:")
    print("- See docs/source/embedding_guide.rst for detailed documentation")
    print("- Run examples/embedding_examples.py for neural network methods")
    print("- Try scripts/embedding/ for configuration-based experiments")


if __name__ == "__main__":
    main()