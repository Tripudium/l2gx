"""
Performance comparison test for geo2 aligner batching.
"""

import copy
import time
import numpy as np
import torch
from l2gx.align.geo2 import GeoAlignmentProblem2
from examples.example import generate_points, voronoi_patches, transform_patches


def test_performance_comparison():
    """Compare performance between different batch sizes."""
    print("=== GEO2 PERFORMANCE COMPARISON ===")
    
    # Generate test data
    np.random.seed(42)
    torch.manual_seed(42)
    
    points = generate_points(n_clusters=3, scale=1.0, std=0.0, max_size=200, min_size=150, dim=2)
    patches, _ = voronoi_patches(points, sample_size=4, min_degree=4, min_overlap=50, min_patch_size=80, eps=1, kmeans=False)
    
    print(f"Generated {len(patches)} patches from {points.shape[0]} points")
    
    # Apply transformations
    np.random.seed(42)
    transformed_patches = transform_patches([copy.deepcopy(p) for p in patches], shift_scale=1.0, scale_range=None)
    
    # Test configurations
    configs = [
        ("Full Batch (Original)", 0),
        ("Batched (512)", 512),
        ("Batched (256)", 256),
        ("Batched (128)", 128),
    ]
    
    results = []
    
    for name, batch_size in configs:
        print(f"\n--- Testing {name} ---")
        
        # Create aligner
        aligner = GeoAlignmentProblem2(
            verbose=False, 
            use_orthogonal_reg=True, 
            orthogonal_reg_weight=10.0,
            batch_size=batch_size
        )
        
        # Time the training
        start_time = time.time()
        
        result = aligner.align_patches(
            patches=[copy.deepcopy(p) for p in transformed_patches],
            num_epochs=100,  # Reduced for timing
            learning_rate=0.01
        )
        
        end_time = time.time()
        
        # Calculate metrics
        training_time = end_time - start_time
        final_loss = result.loss_history[-1]
        embedding = result.get_aligned_embedding()
        
        from scipy.spatial import procrustes
        try:
            _, _, procrustes_error = procrustes(points, embedding)
        except:
            procrustes_error = float('inf')
        
        results.append({
            'name': name,
            'batch_size': batch_size,
            'time': training_time,
            'final_loss': final_loss,
            'procrustes_error': procrustes_error
        })
        
        print(f"Training time: {training_time:.2f}s")
        print(f"Final loss: {final_loss:.6f}")
        print(f"Procrustes error: {procrustes_error:.6f}")
    
    # Performance summary
    print(f"\n=== PERFORMANCE SUMMARY ===")
    baseline_time = results[0]['time']
    
    for result in results:
        speedup = baseline_time / result['time']
        print(f"{result['name']:20} | "
              f"Time: {result['time']:6.2f}s | "
              f"Speedup: {speedup:5.2f}x | "
              f"Loss: {result['final_loss']:8.4f} | "
              f"Procrustes: {result['procrustes_error']:8.6f}")
    
    # Find best performance
    best_time = min(r['time'] for r in results[1:])  # Exclude baseline
    best_config = next(r for r in results if r['time'] == best_time)
    
    print(f"\nBest performance: {best_config['name']} "
          f"({baseline_time/best_time:.2f}x speedup)")


if __name__ == "__main__":
    test_performance_comparison()