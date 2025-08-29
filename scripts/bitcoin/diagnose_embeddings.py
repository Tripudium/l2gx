#!/usr/bin/env python3
"""
Quick diagnostic to check if embeddings are degenerate (constant/collapsed).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from l2gx.datasets import get_dataset
from l2gx.embedding import get_embedding
from l2gx.align import get_aligner


def check_embedding_health(embeddings, method_name):
    """Check if embeddings are healthy or degenerate."""
    print(f"\n{'='*50}")
    print(f"Checking {method_name}")
    print(f"{'='*50}")
    
    # Basic stats
    print(f"Shape: {embeddings.shape}")
    print(f"Overall stats:")
    print(f"  Mean: {np.mean(embeddings):.6f}")
    print(f"  Std:  {np.std(embeddings):.6f}")
    print(f"  Min:  {np.min(embeddings):.6f}")
    print(f"  Max:  {np.max(embeddings):.6f}")
    
    # Check per-dimension variance
    dim_stds = np.std(embeddings, axis=0)
    zero_dims = np.sum(dim_stds < 1e-6)
    low_var_dims = np.sum(dim_stds < 0.01)
    
    print(f"\nPer-dimension analysis:")
    print(f"  Constant dimensions (std < 1e-6): {zero_dims}/{embeddings.shape[1]}")
    print(f"  Low variance dims (std < 0.01):   {low_var_dims}/{embeddings.shape[1]}")
    print(f"  Mean dim std: {np.mean(dim_stds):.6f}")
    print(f"  Min dim std:  {np.min(dim_stds):.6f}")
    print(f"  Max dim std:  {np.max(dim_stds):.6f}")
    
    # Check node similarity
    # Sample some nodes to check if they're all the same
    sample_indices = [0, 100, 200, 300, 400] if len(embeddings) > 400 else range(min(5, len(embeddings)))
    sample_embs = embeddings[sample_indices]
    
    print(f"\nSample node embeddings (first 5 dims):")
    for i, idx in enumerate(sample_indices):
        print(f"  Node {idx}: {sample_embs[i, :5]}")
    
    # Check if all embeddings are identical
    if len(embeddings) > 1:
        all_same = np.allclose(embeddings[0], embeddings, atol=1e-6)
        if all_same:
            print("\n⚠️ WARNING: All embeddings are nearly identical!")
        else:
            # Check pairwise distances
            from scipy.spatial.distance import pdist
            n_sample = min(100, len(embeddings))
            sample_idx = np.random.choice(len(embeddings), n_sample, replace=False)
            distances = pdist(embeddings[sample_idx])
            print(f"\nPairwise distances (sample of {n_sample} nodes):")
            print(f"  Mean: {np.mean(distances):.6f}")
            print(f"  Std:  {np.std(distances):.6f}")
            print(f"  Min:  {np.min(distances):.6f}")
            print(f"  Max:  {np.max(distances):.6f}")
    
    # Health verdict
    print("\n🏥 Health Check:")
    if zero_dims > embeddings.shape[1] * 0.5:
        print("  ❌ UNHEALTHY: Most dimensions have collapsed to constants")
    elif np.std(embeddings) < 0.001:
        print("  ❌ UNHEALTHY: Extremely low overall variance")
    elif low_var_dims > embeddings.shape[1] * 0.8:
        print("  ⚠️ WARNING: Many dimensions have very low variance")
    else:
        print("  ✅ HEALTHY: Embeddings show reasonable variance")
    
    return {
        'std': np.std(embeddings),
        'zero_dims': zero_dims,
        'low_var_dims': low_var_dims,
        'healthy': zero_dims < embeddings.shape[1] * 0.5 and np.std(embeddings) > 0.001
    }


def test_methods():
    """Test different embedding methods."""
    
    print("Loading BTC-reduced dataset...")
    dataset = get_dataset("btc-reduced", max_nodes=2000)
    data = dataset[0]
    print(f"Dataset: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Test dimensions
    test_dims = [32, 64, 128]
    
    methods = {
        "vgae": lambda dim: get_embedding(
            "vgae", embedding_dim=dim, epochs=50, verbose=False
        ),
        "graphsage": lambda dim: get_embedding(
            "graphsage", embedding_dim=dim, epochs=100, learning_rate=0.01, verbose=False
        ),
        "hierarchical": lambda dim: get_embedding(
            "hierarchical", 
            embedding_dim=dim,
            aligner=get_aligner("l2g"),
            max_patch_size=800,
            base_method="vgae",
            epochs=50,
            verbose=False
        )
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"\n{'='*60}")
        print(f"Testing {method_name.upper()}")
        print(f"{'='*60}")
        
        for dim in test_dims:
            print(f"\nDimension {dim}:")
            try:
                embedder = method_func(dim)
                embeddings = embedder.fit_transform(data)
                
                health = check_embedding_health(embeddings, f"{method_name}_dim{dim}")
                
                key = f"{method_name}_{dim}"
                results[key] = health
                
            except Exception as e:
                print(f"  ERROR: {e}")
                results[f"{method_name}_{dim}"] = {'healthy': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("\nHealth Status:")
    print(f"{'Method':<20} {'Dim':<5} {'Status':<10} {'Std':<10} {'Zero Dims'}")
    print("-" * 55)
    
    for key, res in results.items():
        parts = key.rsplit('_', 1)
        method = parts[0]
        dim = parts[1]
        
        if 'error' in res:
            print(f"{method:<20} {dim:<5} {'ERROR':<10}")
        else:
            status = "✅ OK" if res['healthy'] else "❌ BAD"
            std = f"{res['std']:.6f}" if 'std' in res else "N/A"
            zero = f"{res.get('zero_dims', 'N/A')}"
            print(f"{method:<20} {dim:<5} {status:<10} {std:<10} {zero}")


if __name__ == "__main__":
    test_methods()